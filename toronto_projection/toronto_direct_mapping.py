"""
toronto_direct_mapping.py
=========================
Direct mapping of ONT methylation samples to Toronto MG1-4 classification.

Strategy:
  1. Parse GSE180061 EPIC beta matrix (121 reference samples)
  2. Download EPIC array manifest (hg38 coordinates per cg probe)
  3. Select top 10,000 most variable probes (MAD) -- matching paper's method
  4. Train Random Forest classifier on reference samples
  5. Extract ONT methylation at those exact probe positions from bedMethyl files
  6. Predict MG for each of your 14 ONT samples
  7. Plot: UMAP + heatmap of reference + ONT samples colored by MG and S/R

Usage:
  conda activate methylation_plots
  pip install pandas numpy matplotlib seaborn scipy scikit-learn umap-learn requests
  python toronto_direct_mapping.py

Input files (all in input_data/):
  GSE180061_Matrix_processed.txt.gz  -- EPIC beta matrix
  mg_sample_labels.tsv               -- sample -> MG labels
  EPIC_hg38_manifest.tsv.gz          -- EPIC hg38 manifest
"""

import os
import sys
import gzip
import glob
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import median_abs_deviation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import umap

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
MODKIT_DIR  = os.path.join(REPO_DIR, "input_data")
# Directory holding the ONT modkit sample folders (ONTWGS9-<key>-*/3_methylation/*_CpG_5mC.bed)
OUT_BASE    = "/path/to/modkit_output"
OUT_DIR     = os.path.join(REPO_DIR, "output", "toronto_direct_subset")
CACHE_DIR   = os.path.join(REPO_DIR, "cache_toronto_direct")

BETA_FILE   = f"{MODKIT_DIR}/GSE180061_Matrix_processed.txt.gz"
MG_FILE     = f"{MODKIT_DIR}/mg_sample_labels.tsv"
MANIFEST_FILE = f"{MODKIT_DIR}/EPIC_hg38_manifest.tsv.gz"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Top N probes by MAD for classification (matching paper)
TOP_N_PROBES = 10000

# Min coverage in ONT bedMethyl to use a CpG site
MIN_COVERAGE = 5

# Patient map: patient -> (R_key, S_key)
PATIENT_MAP = {
    1: (2,  1),
    3: (4,  5),
    4: (6,  7),
    5: (8,  9),
    6: (10, 11),
    7: (12, 13),
    8: (14, 15),
}

SAMPLE_META = {}
for pat, (rk, sk) in PATIENT_MAP.items():
    SAMPLE_META[rk] = {"label": f"P{pat}R", "patient": pat, "method": "Resection"}
    SAMPLE_META[sk] = {"label": f"P{pat}S", "patient": pat, "method": "Sonopet"}

MG_COLORS = {
    "MG1": "#e74c3c",  # immunogenic -- red
    "MG2": "#3498db",  # NF2-wildtype -- blue
    "MG3": "#2ecc71",  # hypermetabolic -- green
    "MG4": "#9b59b6",  # proliferative -- purple
}
METHOD_COLORS = {"Sonopet": "#5b9bd5", "Resection": "#ed7d31"}
MG_LABELS = {
    "MG1": "MG1 Immunogenic",
    "MG2": "MG2 NF2-wildtype",
    "MG3": "MG3 Hypermetabolic",
    "MG4": "MG4 Proliferative",
}

# ─── PLOT CONFIG ──────────────────────────────────────────────────────────────
PLOT_CFG = dict(
    fontsize_title   = 20,
    fontsize_label   = 15,   # axis labels
    fontsize_tick    = 15,    # tick labels
    fontsize_legend  = 15,    # legend text
    fontsize_annot   = 15,    # in-plot sample label annotations
    fontsize_cbar    = 15,    # colorbar label + ticks
    fontfamily       = "monospace",
)

# ─── STEP 1: DOWNLOAD EPIC MANIFEST ──────────────────────────────────────────

def download_manifest():
    """
    Download EPIC array hg38 manifest from Illumina / GitHub.
    Columns we need: Name (cg ID), CHR, MAPINFO (position)
    """
    if os.path.exists(MANIFEST_FILE):
        print("[INFO] Manifest already exists, skipping download")
        return

    print("[INFO] Downloading EPIC array hg38 manifest...")
    # Use the minfi-compatible manifest from BioConductor annotation package
    # hosted on GitHub (IlluminaHumanMethylationEPICanno.ilm10b4.hg19 -> hg38 liftover)
    # We use the Bret Barnes / Zhou lab hg38 manifest which is widely used
    url = "https://zhouserver.research.chop.edu/InfiniumAnnotation/20180909/EPIC/EPIC.hg38.manifest.tsv.gz"
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(MANIFEST_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Manifest saved: {MANIFEST_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not download manifest: {e}")
        print("  Please download manually and save to:")
        print(f"  {MANIFEST_FILE}")
        print("  URL: https://zhouserver.research.chop.edu/InfiniumAnnotation/20180909/EPIC/EPIC.hg38.manifest.tsv.gz")
        sys.exit(1)


def load_manifest():
    """Load probe -> (chrom, pos) mapping from manifest."""
    print("[INFO] Loading EPIC manifest...")
    df = pd.read_csv(MANIFEST_FILE, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"],
                     low_memory=False)
    df = df.dropna(subset=["CpG_chrm", "CpG_beg"])
    df["CpG_beg"] = df["CpG_beg"].astype(int)
    # Keep canonical chromosomes only
    df = df[df["CpG_chrm"].str.match(r"^chr([0-9]+|X|Y)$")]
    df = df.set_index("probeID")
    print(f"[INFO] {len(df):,} probes with hg38 coordinates")
    return df


# ─── STEP 2: PARSE BETA MATRIX ────────────────────────────────────────────────

def load_beta_matrix():
    """
    Parse GSE180061_Matrix_processed.txt.gz.
    Returns DataFrame: probes x samples (beta values only, detection p-vals dropped).
    """
    cache = f"{CACHE_DIR}/beta_matrix.pkl"
    if os.path.exists(cache):
        print("[INFO] Loading cached beta matrix...")
        return pd.read_pickle(cache)

    print("[INFO] Parsing beta matrix (this may take a few minutes)...")

    # Parse line by line to avoid pandas header/index confusion.
    # Header row: sample1 \t DetPval \t sample2 \t DetPval ...  (N*2 fields)
    # Data rows:  cg_id   \t beta1   \t pval1   \t beta2  ...  (1 + N*2 fields)
    # The probe ID field has no corresponding header entry.

    data = {}
    with gzip.open(BETA_FILE, "rt") as fh:
        # Parse header
        raw_header = fh.readline().strip().split("\t")
        raw_header = [h.strip('"') for h in raw_header]
        # Even indices = sample names, odd indices = "Detection Pval"
        sample_names = [h for i, h in enumerate(raw_header) if i % 2 == 0]
        beta_indices = [i + 1 for i in range(0, len(raw_header), 2)]  # 1-based in data row

        # Parse data rows
        for line in fh:
            fields = line.split("\t")
            probe_id = fields[0].strip('"')
            try:
                betas = [float(fields[i]) for i in beta_indices]
            except (ValueError, IndexError):
                continue
            data[probe_id] = betas

    df = pd.DataFrame.from_dict(data, orient="index", columns=sample_names)

    print(f"[INFO] Beta matrix: {df.shape[0]:,} probes x {df.shape[1]} samples")
    df.to_pickle(cache)
    return df


# ─── STEP 3: SELECT TOP VARIABLE PROBES ───────────────────────────────────────

def select_top_probes(beta_df, mg_labels, manifest_df):
    """Select top N probes by MAD, filtered to those with hg38 coordinates."""
    cache = f"{CACHE_DIR}/top_probes.pkl"
    if os.path.exists(cache):
        print("[INFO] Loading cached top probes...")
        return pd.read_pickle(cache)

    print(f"[INFO] Selecting top {TOP_N_PROBES} probes by MAD...")

    # Keep only samples with MG labels
    ref_samples = [s for s in beta_df.columns if s in mg_labels.index]
    beta_ref = beta_df[ref_samples].copy()

    # Drop probes with too many NaNs
    beta_ref = beta_ref.dropna(thresh=int(len(ref_samples) * 0.8))

    # Keep only probes with hg38 coordinates
    common_probes = beta_ref.index.intersection(manifest_df.index)
    beta_ref = beta_ref.loc[common_probes]
    print(f"[INFO] Probes with coordinates: {len(beta_ref):,}")

    # Compute MAD per probe
    mad = beta_ref.apply(lambda row: median_abs_deviation(row.dropna()), axis=1)
    top_probes = mad.nlargest(TOP_N_PROBES).index

    result = beta_ref.loc[top_probes]
    result.to_pickle(cache)
    print(f"[INFO] Top probes selected: {result.shape}")
    return result


# ─── STEP 4: TRAIN RANDOM FOREST CLASSIFIER ───────────────────────────────────

def train_classifier(beta_top, mg_labels):
    """Train Random Forest on reference samples."""
    print("[INFO] Training Random Forest classifier...")

    ref_samples = [s for s in beta_top.columns if s in mg_labels.index]
    X = beta_top[ref_samples].T.fillna(0.5).values  # samples x probes
    y = mg_labels.loc[ref_samples, "mg_num"].values

    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X, y)

    # Quick in-sample report (just for sanity check)
    y_pred = clf.predict(X)
    print("[INFO] In-sample classification report (reference cohort):")
    print(classification_report(y, y_pred, target_names=["MG1","MG2","MG3","MG4"]))

    return clf, ref_samples


# ─── STEP 5: EXTRACT ONT METHYLATION AT PROBE POSITIONS ──────────────────────

def find_bedmethyl(key):
    pattern = f"{OUT_BASE}/ONTWGS9-{key}-*"
    matches = glob.glob(pattern)
    if not matches:
        return None
    sample_id = os.path.basename(matches[0])
    bed = f"{OUT_BASE}/{sample_id}/3_methylation/{sample_id}_CpG_5mC.bed"
    return bed if os.path.exists(bed) else None


def load_ont_sample(bed_path, probe_positions):
    """
    Load ONT methylation at specific probe positions.
    probe_positions: dict of chrom -> {pos -> probe_id}
    Returns Series: probe_id -> methylation fraction (0-1)
    """
    values = {}
    opener = gzip.open if bed_path.endswith(".gz") else open

    with opener(bed_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 11:
                continue
            chrom = f[0]
            pos   = int(f[1])
            if chrom not in probe_positions:
                continue
            if pos not in probe_positions[chrom]:
                continue
            try:
                coverage = int(f[9])
                freq     = float(f[10])
            except ValueError:
                continue
            if coverage < MIN_COVERAGE:
                continue
            probe_id = probe_positions[chrom][pos]
            values[probe_id] = freq / 100.0

    return pd.Series(values)


def extract_ont_matrix(top_probes_df, manifest_df):
    """Extract ONT methylation for all samples at top probe positions."""
    cache = f"{CACHE_DIR}/ont_matrix.pkl"
    if os.path.exists(cache):
        print("[INFO] Loading cached ONT matrix...")
        return pd.read_pickle(cache)

    print("[INFO] Building probe position lookup...")
    # Build chrom -> pos -> probe_id lookup for top probes only
    top_manifest = manifest_df.loc[manifest_df.index.intersection(top_probes_df.index)]
    probe_positions = {}
    for probe_id, row in top_manifest.iterrows():
        chrom = row["CpG_chrm"]
        pos   = row["CpG_beg"]
        if chrom not in probe_positions:
            probe_positions[chrom] = {}
        probe_positions[chrom][pos] = probe_id

    print(f"[INFO] Lookup covers {sum(len(v) for v in probe_positions.values()):,} positions")

    series_list = []
    for key, meta in sorted(SAMPLE_META.items()):
        bed = find_bedmethyl(key)
        if bed is None:
            print(f"[WARN] bedMethyl not found for key {key} ({meta['label']}), skipping")
            continue
        print(f"  [ONT] Extracting {meta['label']} (key {key})...")
        s = load_ont_sample(bed, probe_positions).rename(meta["label"])
        series_list.append(s)

    mat = pd.concat(series_list, axis=1)
    mat.to_pickle(cache)
    print(f"[INFO] ONT matrix: {mat.shape[0]:,} probes x {mat.shape[1]} samples")
    return mat


# ─── STEP 6: PREDICT MG FOR ONT SAMPLES ──────────────────────────────────────

def predict_ont(clf, ont_matrix, top_probes_df):
    """Predict MG for each ONT sample."""
    print("[INFO] Predicting MG for ONT samples...")

    # Align ONT matrix to top probe order, fill missing with 0.5 (neutral)
    X_ont = ont_matrix.reindex(top_probes_df.index).fillna(0.5).T.values

    predictions  = clf.predict(X_ont)
    probabilities = clf.predict_proba(X_ont)

    results = pd.DataFrame({
        "sample":    ont_matrix.columns,
        "predicted": predictions,
        "prob_MG1":  probabilities[:, 0],
        "prob_MG2":  probabilities[:, 1],
        "prob_MG3":  probabilities[:, 2],
        "prob_MG4":  probabilities[:, 3],
    })

    print("\n[RESULTS] ONT sample MG predictions:")
    print(f"{'Sample':<10} {'Predicted':>12} {'MG1':>6} {'MG2':>6} {'MG3':>6} {'MG4':>6}")
    print("-" * 50)
    for _, row in results.iterrows():
        print(f"{row['sample']:<10} {row['predicted']:>12} "
              f"{row['prob_MG1']:>6.2f} {row['prob_MG2']:>6.2f} "
              f"{row['prob_MG3']:>6.2f} {row['prob_MG4']:>6.2f}")

    results.to_csv(f"{OUT_DIR}/ont_mg_predictions.tsv", sep="\t", index=False)
    print(f"\n[SAVED] {OUT_DIR}/ont_mg_predictions.tsv")
    return results


# ─── STEP 7: UMAP ─────────────────────────────────────────────────────────────

def plot_umap(top_probes_df, mg_labels, ont_matrix, predictions):
    print("[INFO] Running UMAP...")

    ref_samples = [s for s in top_probes_df.columns if s in mg_labels.index]
    ont_samples = list(ont_matrix.columns)

    # Build combined matrix: reference + ONT
    ref_mat = top_probes_df[ref_samples].fillna(0.5)
    ont_mat = ont_matrix.reindex(top_probes_df.index).fillna(0.5)

    combined = pd.concat([ref_mat, ont_mat], axis=1).T  # samples x probes

    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=42)
    embedding = reducer.fit_transform(combined.values)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    # Plot reference samples (triangles, small, semi-transparent)
    for i, sample in enumerate(ref_samples):
        mg  = mg_labels.loc[sample, "mg_num"]
        col = MG_COLORS.get(mg, "#aaaaaa")
        ax.scatter(embedding[i, 0], embedding[i, 1],
                   c=col, s=35, alpha=0.35, marker="^",
                   edgecolors="none", zorder=2)

    # Plot study samples (large, with labels)
    pred_dict = dict(zip(predictions["sample"], predictions["predicted"]))
    for i, sample in enumerate(ont_samples):
        idx    = len(ref_samples) + i
        mg     = pred_dict.get(sample, "unknown")
        col    = MG_COLORS.get(mg, "#aaaaaa")
        meta   = next(m for m in SAMPLE_META.values() if m["label"] == sample)
        marker = "o" if meta["method"] == "Sonopet" else "s"
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=col, s=180, marker=marker,
                   edgecolors="#222222", linewidth=1.2, zorder=4)
        ax.annotate(sample,
                    (embedding[idx, 0], embedding[idx, 1]),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=PLOT_CFG["fontsize_annot"],
                    fontfamily=PLOT_CFG["fontfamily"],
                    color="#222222", zorder=5)

    # Connect S-R pairs with dashed lines
    for pat, (rk, sk) in PATIENT_MAP.items():
        r_label = f"P{pat}R"
        s_label = f"P{pat}S"
        if r_label in ont_samples and s_label in ont_samples:
            ri = len(ref_samples) + ont_samples.index(r_label)
            si = len(ref_samples) + ont_samples.index(s_label)
            ax.plot([embedding[ri,0], embedding[si,0]],
                    [embedding[ri,1], embedding[si,1]],
                    color="#555555", linewidth=1, alpha=0.6,
                    linestyle="--", zorder=3)

    # Single unified legend
    legend_handles = (
        [mpatches.Patch(color=c, label=MG_LABELS[mg]) for mg, c in MG_COLORS.items()]
        + [mpatches.Patch(color="none", label="")]  # spacer
        + [
            plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555555",
                       markeredgecolor="#222222", markersize=9, label="Sonopet"),
            plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555555",
                       markeredgecolor="#222222", markersize=9, label="Resection"),
            plt.Line2D([0],[0], marker="^", color="w", markerfacecolor="#aaaaaa",
                       markersize=8, alpha=0.6, label=f"Reference (n={len(ref_samples)})"),
            plt.Line2D([0],[0], color="#555555", linewidth=1.2, linestyle="--",
                       label="Matched S–R pair"),
        ]
    )
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=PLOT_CFG["fontsize_legend"],
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel("UMAP 1",
                  fontsize=PLOT_CFG["fontsize_label"],
                  fontfamily=PLOT_CFG["fontfamily"])
    ax.set_ylabel("UMAP 2",
                  fontsize=PLOT_CFG["fontsize_label"],
                  fontfamily=PLOT_CFG["fontfamily"])
    ax.tick_params(axis="both", labelsize=PLOT_CFG["fontsize_tick"])
    ax.set_title(
        "Toronto molecular group classification",
        fontsize=PLOT_CFG["fontsize_title"], fontweight="bold",
        fontfamily=PLOT_CFG["fontfamily"], pad=10
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#000000")


    # ── Reference-only UMAP sanity check ──────────────────────────────────────
    reducer_ref = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=42)
    emb_ref = reducer_ref.fit_transform(top_probes_df[ref_samples].fillna(0.5).T.values)

    fig2, ax2 = plt.subplots(figsize=(8, 7), facecolor="white")
    ax2.set_facecolor("white")
    for i, s in enumerate(ref_samples):
        mg  = mg_labels.loc[s, "mg_num"]
        col = MG_COLORS.get(mg, "#aaaaaa")
        ax2.scatter(emb_ref[i,0], emb_ref[i,1], c=col, s=40, alpha=0.7,
                    marker="o", edgecolors="none")
    import matplotlib.patches as mp2
    handles = [mp2.Patch(color=c, label=MG_LABELS[mg])
               for mg, c in MG_COLORS.items()]
    ax2.legend(handles=handles, fontsize=8)
    ax2.set_xlabel("UMAP 1", fontsize=10, fontfamily="monospace")
    ax2.set_ylabel("UMAP 2", fontsize=10, fontfamily="monospace")
    ax2.set_title("Reference cohort UMAP (n=121, no ONT samples)",
                  fontsize=11, fontweight="bold", fontfamily="monospace")
    out2 = f"{OUT_DIR}/umap_reference_only.pdf"
    fig2.savefig(out2, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[SAVED] {out2}")


    out = f"{OUT_DIR}/toronto_umap.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ─── STEP 8: CONCORDANCE HEATMAP ─────────────────────────────────────────────

def plot_concordance(predictions):
    """
    Paired concordance heatmap: for each patient, S and R are adjacent rows.
    Shows MG classification probability per sample, box marks predicted MG.
    """
    print("[INFO] Plotting concordance heatmap...")

    pred = predictions.set_index("sample")
    prob_cols = ["prob_MG1", "prob_MG2", "prob_MG3", "prob_MG4"]
    prob_mat  = pred[prob_cols].copy()
    prob_mat.columns = ["MG1\nImmunogenic", "MG2\nNF2-wt",
                        "MG3\nHypermet.", "MG4\nProlif."]

    # Paired order: P1S, P1R, P3S, P3R, ... so each patient pair is adjacent
    paired_order = []
    patient_boundaries = []  # row indices where patient blocks end
    for p in sorted(PATIENT_MAP):
        s_lbl = f"P{p}S"
        r_lbl = f"P{p}R"
        pair = [l for l in [s_lbl, r_lbl] if l in prob_mat.index]
        patient_boundaries.append(len(paired_order) + len(pair) - 0.5)
        paired_order.extend(pair)

    prob_mat = prob_mat.loc[paired_order]

    # Y tick labels colored by method
    ytick_labels = list(prob_mat.index)
    ytick_colors = [METHOD_COLORS["Sonopet"] if l.endswith("S")
                    else METHOD_COLORS["Resection"] for l in ytick_labels]

    fig, ax = plt.subplots(figsize=(6, 9), facecolor="white")
    ax.set_facecolor("white")

    im = ax.imshow(prob_mat.values, aspect="auto",
                   cmap="YlOrRd", vmin=0, vmax=1)

    # X axis
    ax.set_xticks(range(4))
    ax.set_xticklabels(prob_mat.columns,
                       fontsize=PLOT_CFG["fontsize_tick"],
                       fontfamily=PLOT_CFG["fontfamily"], ha="center")
    ax.tick_params(axis="x", pad=8)

    # Y axis — patient labels
    ax.set_yticks(range(len(prob_mat)))
    ax.set_yticklabels(ytick_labels,
                       fontsize=PLOT_CFG["fontsize_tick"],
                       fontfamily=PLOT_CFG["fontfamily"])
    for tick, color in zip(ax.get_yticklabels(), ytick_colors):
        tick.set_color(color)
    ax.tick_params(axis="y", pad=6)

    # Box around predicted MG
    for i, sample in enumerate(prob_mat.index):
        mg = pred.loc[sample, "predicted"]
        j  = ["MG1", "MG2", "MG3", "MG4"].index(mg)
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#111111", linewidth=2.5
        ))

    # Thin lines between patient pairs
    for boundary in patient_boundaries[:-1]:
        ax.axhline(boundary, color="#888888", linewidth=0.8, linestyle="--")

    # Method legend as colored text outside plot
    fig.text(0.01, 0.5, "Sonopet", va="center", ha="left",
             fontsize=PLOT_CFG["fontsize_tick"],
             fontfamily=PLOT_CFG["fontfamily"],
             color=METHOD_COLORS["Sonopet"], fontweight="bold", rotation=90,
             transform=fig.transFigure)
    fig.text(0.04, 0.5, "Resection", va="center", ha="left",
             fontsize=PLOT_CFG["fontsize_tick"],
             fontfamily=PLOT_CFG["fontfamily"],
             color=METHOD_COLORS["Resection"], fontweight="bold", rotation=90,
             transform=fig.transFigure)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Probability",
                   fontsize=PLOT_CFG["fontsize_cbar"],
                   fontfamily=PLOT_CFG["fontfamily"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(),
             fontsize=PLOT_CFG["fontsize_cbar"],
             fontfamily=PLOT_CFG["fontfamily"])

    ax.set_title(
        "Toronto molecular group classification\nper-patient concordance",
        fontsize=PLOT_CFG["fontsize_title"], fontweight="bold",
        fontfamily=PLOT_CFG["fontfamily"], pad=12
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    out = f"{OUT_DIR}/toronto_concordance.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Manifest
    download_manifest()
    manifest_df = load_manifest()

    # Step 2: Beta matrix
    beta_df = load_beta_matrix()

    # Step 3: MG labels
    mg_labels = pd.read_csv(MG_FILE, sep="\t", index_col=0)
    print(f"[INFO] MG labels loaded: {len(mg_labels)} samples")

    # Step 4: Top probes
    top_probes_df = select_top_probes(beta_df, mg_labels, manifest_df)

    # Step 5: Train classifier
    clf, ref_samples = train_classifier(top_probes_df, mg_labels)

    # Step 6: Extract ONT methylation at probe positions
    ont_matrix = extract_ont_matrix(top_probes_df, manifest_df)

    # Step 7: Predict
    predictions = predict_ont(clf, ont_matrix, top_probes_df)

    # Step 8: Plots
    plot_umap(top_probes_df, mg_labels, ont_matrix, predictions)
    plot_concordance(predictions)

    print("\nDone.")
