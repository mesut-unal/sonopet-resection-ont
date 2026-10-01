"""
toronto_projection.py
=====================
Project 12 ONT methylation samples onto the Toronto reference cohort (121 EPIC).

Runs all 4 combinations of:
  - normalization : {quantile, mean_center}    (within-platform, separately)
  - feature space : {probe, cpg_island_rank}

Pipeline per combination:
  1. Intersect Toronto + ONT on probe ID, drop probes with any NaN.
  2. Within-platform normalization (Toronto and ONT separately).
  3. Per-sample z-score across probes.
  4. Optional feature transform (CpG island rank).
  5. Pearson correlation: ONT × Toronto.
  6. MG assignment per ONT sample via mean correlation to each Toronto MG.
  7. Outputs:
       - projection heatmap   (12 ONT × 121 Toronto, ordered by predicted MG / dendrogram)
       - combined consensus   (133 × 133, dendrogram + cluster strip + ONT marker strip)
       - MG assignments TSV   (sample → MG, top-1 corr, top-2 MG, margin)

Usage:
  conda activate methylation_plots
  python toronto_projection.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

# ---- paths ------------------------------------------------------------------
# Repo root = directory containing this script
REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(REPO_DIR, "data")
INPUT_DIR = os.path.join(REPO_DIR, "input_data")

# Large inputs stay local (not in repo — see README)
LOCAL_DIR = os.path.join(REPO_DIR, "cache_toronto_direct_7patients")
ANNOT_DIR = INPUT_DIR

TORONTO_PKL           = os.path.join(LOCAL_DIR, "top_probes.pkl")
ONT_PKL               = os.path.join(LOCAL_DIR, "ont_matrix.pkl")
TORONTO_LBL           = os.path.join(DATA_DIR,  "consensus_labels_k6.pkl")
CACHE_DIR             = DATA_DIR   # probe_to_island.pkl, probe_to_chrom.pkl cached here
CACHE_PROBE_TO_ISLAND = os.path.join(DATA_DIR, "probe_to_island.pkl")
CACHE_PROBE_TO_CHROM  = os.path.join(DATA_DIR, "probe_to_chrom.pkl")

_RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(REPO_DIR, "output",
                          f"toronto_projection_7patients_{_RUN_STAMP}")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

K               = 6
DROP_SEX_CHROMS = True   # drop chrX/chrY probes before any analysis

EPIC_MANIFEST_FILE = os.path.join(ANNOT_DIR, "EPIC_hg38_manifest.tsv.gz")
UCSC_CPGI_FILE     = os.path.join(ANNOT_DIR, "cpgIslandExt.txt.gz")

# ---- colors -----------------------------------------------------------------
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "white_purple",
    ["#ffffff","#eee8f5","#c9b8e8","#9b72cf","#6a3d9a","#3d1a6e"])

CLUSTER_COLORS = {
    1:"#f5e642", 2:"#e8c84a", 3:"#d4a843",
    4:"#c07a2e", 5:"#a04e1a", 6:"#6b2a0a",
}
ONT_MARKER_COLOR = "#1f77b4"   # blue stripe for ONT samples in combined plot

# ---- font sizes -------------------------------------------------------------
FONT = {
    "suptitle"          : 20,   # large suptitle (combined/MG plots)
    "suptitle_medium"   : 13,   # _set_title default
    "suptitle_small"    : 11,   # projection heatmap suptitle
    "axis_label"        : 15,    # x/y axis labels
    "ticklabel"         : 15,    # general tick labels
    "ticklabel_small"   : 7,    # colorbar tick params
    "cell_annotation"   : 15,    # in-cell value text
    "sample_label"      : 15,    # ONT sample name annotations on axes
    "legend"            : 15,   # legend entries (combined plots)
    "legend_title"      : 15,   # legend titles (combined plots)
    "legend_small"      : 8,    # legend entries (projection / toronto-only)
    "legend_title_small": 8,    # legend titles (projection / toronto-only)
    "cbar_label"        : 15,    # colorbar label (most plots)
    "cbar_label_large"  : 15,   # colorbar label in combined heatmap
    "cbar_label_small"  : 15,    # colorbar label (prob heatmap / projection)
}

# Light→dark blue gradient for predicted MG (used in ONT marker strips)
MG_BLUE_SHADES = {
    "MG1": "#c6dbef",   # lightest blue
    "MG2": "#6baed6",
    "MG3": "#2171b5",
    "MG4": "#08306b",   # darkest blue
}
MG_UNKNOWN_COLOR = "#cccccc"   # used when prediction is missing/uncertain


# =============================================================================
# Data loading
# =============================================================================
def generate_consensus_labels_if_missing(n_iter=1000, subsample_frac=0.80, k=K,
                                          seed=42):
    """If consensus_labels_k6.pkl is missing, generate it from scratch using the
    same procedure as the original toronto_projection_heatmap.py:
      - raw top_probes.pkl (NaN→0.5, no normalization, no sex-chrom filter)
      - 1000 iterations of 80% subsampled Ward+fcluster at k=6
      - Build consensus matrix, run final Ward+fcluster on it → integer labels
    Saves both consensus_matrix_k6.pkl and consensus_labels_k6.pkl, then returns.
    No-op if the labels file already exists."""
    if os.path.exists(TORONTO_LBL):
        return

    print(f"[INFO] {TORONTO_LBL} not found — generating from scratch")
    print(f"       (1000 subsampled clustering runs, raw top_probes.pkl)")

    toronto_raw = pd.read_pickle(TORONTO_PKL)        # 10000 × 121
    toronto_raw = toronto_raw.fillna(0.5)            # match old script's NaN handling
    samples = toronto_raw.columns.tolist()
    n_samples = len(samples)
    mat_sf = toronto_raw.values.T                    # (n_samples, n_features)

    co_occur = np.zeros((n_samples, n_samples), dtype=np.float32)
    selected = np.zeros((n_samples, n_samples), dtype=np.float32)

    rng = np.random.default_rng(seed=seed)
    n_sub = int(n_samples * subsample_frac)
    for it in range(n_iter):
        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1}/{n_iter}")
        sub_idx = np.sort(rng.choice(n_samples, n_sub, replace=False))
        sel_outer = np.ix_(sub_idx, sub_idx)
        selected[sel_outer] += 1
        sub_mat = mat_sf[sub_idx, :]
        dist = np.clip(1 - np.corrcoef(sub_mat), 0, 2)
        np.fill_diagonal(dist, 0)
        Z_it = linkage(squareform(dist, checks=False), method="ward")
        labels_it = fcluster(Z_it, k, criterion="maxclust")
        for cl in np.unique(labels_it):
            members = sub_idx[labels_it == cl]
            co_occur[np.ix_(members, members)] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(selected > 0, co_occur / selected, 0.0)
    np.fill_diagonal(consensus, 1.0)
    consensus_df = pd.DataFrame(consensus, index=samples, columns=samples)
    consensus_df.to_pickle(f"{CACHE_DIR}/consensus_matrix_k{k}.pkl")

    # Final cluster call on the consensus matrix
    dist_f = squareform(np.clip(1 - consensus, 0, 2), checks=False)
    Z_f = linkage(dist_f, method="ward")
    final_labels = fcluster(Z_f, k, criterion="maxclust")
    labels_series = pd.Series(final_labels, index=samples, name="cluster")
    labels_series.to_pickle(TORONTO_LBL)
    print(f"[SAVED] {CACHE_DIR}/consensus_matrix_k{k}.pkl")
    print(f"[SAVED] {TORONTO_LBL}")
    print(f"        Cluster sizes: {labels_series.value_counts().sort_index().to_dict()}")


def load_inputs():
    generate_consensus_labels_if_missing()
    toronto = pd.read_pickle(TORONTO_PKL)
    ont     = pd.read_pickle(ONT_PKL)
    lbl     = pd.read_pickle(TORONTO_LBL)
    mg4     = pd.read_csv(f"{INPUT_DIR}/mg_sample_labels.tsv",
                          sep="\t", index_col=0)["mg_num"]

    common = toronto.index.intersection(ont.index)
    toronto = toronto.loc[common]
    ont     = ont.loc[common]

    # Drop probes with any NaN across either cohort
    keep = ~(toronto.isna().any(axis=1) | ont.isna().any(axis=1))
    toronto = toronto.loc[keep]
    ont     = ont.loc[keep]

    if DROP_SEX_CHROMS:
        toronto, ont = drop_sex_chromosome_probes(toronto, ont)

    print(f"[INFO] Toronto: {toronto.shape}, ONT: {ont.shape}, "
          f"MG labels: {len(lbl)}, 4-MG labels: {len(mg4)}")
    return toronto, ont, lbl, mg4


# =============================================================================
# Annotation loading (local files, no network)
# =============================================================================
def load_probe_to_island_map(probe_ids):
    """Return Series: probe_id → island_id (chr:start-end), NaN if not in any island."""
    if os.path.exists(CACHE_PROBE_TO_ISLAND):
        m = pd.read_pickle(CACHE_PROBE_TO_ISLAND)
        return m.reindex(probe_ids)

    for f in (EPIC_MANIFEST_FILE, UCSC_CPGI_FILE):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Annotation file not found: {f}")

    print("[INFO] Parsing EPIC manifest")
    man = pd.read_csv(EPIC_MANIFEST_FILE, sep="\t", compression="gzip",
                      usecols=["probeID", "CpG_chrm", "CpG_beg", "CpG_end"],
                      low_memory=False)
    man = man.dropna(subset=["CpG_chrm", "CpG_beg"])
    man["CpG_beg"] = man["CpG_beg"].astype(int)
    man["CpG_end"] = man["CpG_end"].astype(int)

    print("[INFO] Parsing UCSC cpgIslandExt")
    cpgi = pd.read_csv(UCSC_CPGI_FILE, sep="\t", compression="gzip",
                       header=None, usecols=[1, 2, 3],
                       names=["chrm", "start", "end"])
    cpgi["island_id"] = (cpgi["chrm"] + ":"
                         + cpgi["start"].astype(str) + "-"
                         + cpgi["end"].astype(str))

    print("[INFO] Mapping probes to islands")
    out = {}
    for chrm, g_man in man.groupby("CpG_chrm"):
        g_cpgi = cpgi[cpgi["chrm"] == chrm]
        if g_cpgi.empty:
            continue
        starts = g_cpgi["start"].values
        ends   = g_cpgi["end"].values
        ids    = g_cpgi["island_id"].values
        probe_pos = g_man["CpG_beg"].values
        probe_ids_chrm = g_man["probeID"].values
        idx_sorted = np.argsort(starts)
        starts_s, ends_s, ids_s = starts[idx_sorted], ends[idx_sorted], ids[idx_sorted]
        i = np.searchsorted(starts_s, probe_pos, side="right") - 1
        valid = (i >= 0) & (probe_pos < ends_s[np.clip(i, 0, len(ends_s)-1)])
        for j, ok in enumerate(valid):
            if ok:
                out[probe_ids_chrm[j]] = ids_s[i[j]]

    m = pd.Series(out, name="island_id")
    m.to_pickle(CACHE_PROBE_TO_ISLAND)
    print(f"[INFO] {len(m)} probes mapped to {m.nunique()} islands")
    print(f"[SAVED] {CACHE_PROBE_TO_ISLAND}")
    return m.reindex(probe_ids)


def load_probe_to_chrom_map(probe_ids):
    """Return Series: probe_id → chromosome (e.g. 'chr1', 'chrX')."""
    if os.path.exists(CACHE_PROBE_TO_CHROM):
        m = pd.read_pickle(CACHE_PROBE_TO_CHROM)
        return m.reindex(probe_ids)

    if not os.path.exists(EPIC_MANIFEST_FILE):
        raise FileNotFoundError(f"EPIC manifest not found: {EPIC_MANIFEST_FILE}")

    print("[INFO] Parsing EPIC manifest for chromosome map")
    man = pd.read_csv(EPIC_MANIFEST_FILE, sep="\t", compression="gzip",
                      usecols=["probeID", "CpG_chrm"], low_memory=False)
    man = man.dropna(subset=["CpG_chrm"])
    m = pd.Series(man["CpG_chrm"].values, index=man["probeID"].values,
                  name="chrom")
    m.to_pickle(CACHE_PROBE_TO_CHROM)
    print(f"[INFO] {len(m)} probes mapped to chromosomes; cached")
    return m.reindex(probe_ids)


def drop_sex_chromosome_probes(toronto, ont):
    """Filter both matrices to autosomal probes only (drop chrX, chrY)."""
    chrom = load_probe_to_chrom_map(toronto.index)
    sex_mask = chrom.isin(["chrX", "chrY"])
    n_drop = int(sex_mask.sum())
    print(f"[INFO] Dropping {n_drop} sex-chromosome probes "
          f"(chrX={(chrom == 'chrX').sum()}, chrY={(chrom == 'chrY').sum()})")
    keep_idx = chrom.index[~sex_mask & chrom.notna()]
    keep_idx = keep_idx.intersection(toronto.index).intersection(ont.index)
    return toronto.loc[keep_idx], ont.loc[keep_idx]


# =============================================================================
# Normalization
# =============================================================================
def quantile_normalize(df):
    """Quantile-normalize columns (samples) within a single matrix."""
    ranks = df.apply(lambda s: rankdata(s, method="average"), axis=0)
    sorted_vals = np.sort(df.values, axis=0)
    means = sorted_vals.mean(axis=1)
    # ranks are 1..n; map each rank to mean value at that rank
    out = pd.DataFrame(
        means[np.clip(ranks.values.astype(int) - 1, 0, len(means) - 1)],
        index=df.index, columns=df.columns)
    return out

def mean_center(df):
    """Subtract per-sample mean."""
    return df.sub(df.mean(axis=0), axis=1)

def zscore_per_sample(df):
    """Z-score each column across rows."""
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, 1)
    return df.sub(mu, axis=1).div(sd, axis=1)


# =============================================================================
# Feature transform
# =============================================================================
def aggregate_to_cpg_island_rank(df, probe_to_island):
    """Aggregate probes → islands (mean β within island), then per-sample rank."""
    df_with = df.assign(_island=probe_to_island.reindex(df.index).values)
    df_with = df_with.dropna(subset=["_island"])
    grouped = df_with.groupby("_island").mean()
    # Per-sample rank across islands → scale to [0,1]
    ranked = grouped.apply(lambda s: rankdata(s, method="average") / len(s), axis=0)
    return ranked


def predict_mg_from_clusters(assignments, lbl, mg_labels, tag):
    """Map ONT cluster assignments → paper's 4-MG labels via the 121 Toronto
    samples as a bridge.

    For each of the 6 clusters, compute what fraction of Toronto samples in
    that cluster belong to each MG (MG1-MG4). Then for each ONT sample,
    look up its assigned cluster and report:
      - predicted_MG : hard call (majority vote)
      - MG1..MG4     : probability (fraction of Toronto samples in that cluster)
      - cluster_purity: fraction of the assigned cluster belonging to predicted MG
      - n_toronto_in_cluster: how many Toronto samples support this cluster

    Also saves:
      - mg4_predictions_{tag}.tsv
      - mg4_probability_heatmap_{tag}.png  (ONT samples × MG1-4, colored by prob)
      - cluster_to_mg_contingency_{tag}.tsv
    """
    # ── Build contingency table ───────────────────────────────────────────────
    # Align lbl (6-cluster) and mg_labels (4-MG) on shared sample IDs
    common = lbl.index.intersection(mg_labels.index)
    ct = pd.crosstab(
        lbl.loc[common].rename("cluster"),
        mg_labels.loc[common].rename("MG"),
    )
    ct.to_csv(f"{OUT_DIR}/cluster_to_mg_contingency_{tag}.tsv", sep="\t")
    print(f"[SAVED] cluster_to_mg_contingency_{tag}.tsv")
    print(f"\n[INFO] Cluster → MG contingency:\n{ct.to_string()}\n")

    # Cluster → MG probability distribution (row-normalize contingency)
    ct_prob = ct.div(ct.sum(axis=1), axis=0)   # each row sums to 1

    # MG columns in a fixed order
    mg_cols = sorted(ct_prob.columns.tolist())  # ['MG1','MG2','MG3','MG4']

    # ── Predict per ONT sample ────────────────────────────────────────────────
    rows = []
    for sample, row in assignments.iterrows():
        cl = int(row["assigned_MG"])  # cluster ID from 6-cluster step
        if cl not in ct_prob.index:
            # cluster has no Toronto samples — can't predict
            rows.append({
                "sample"            : sample,
                "assigned_cluster"  : cl,
                "predicted_MG"      : "unknown",
                "cluster_purity"    : np.nan,
                "n_toronto_in_cluster": 0,
                **{mg: np.nan for mg in mg_cols},
            })
            continue
        probs = ct_prob.loc[cl]          # Series: MG1..MG4 probabilities
        pred_mg = probs.idxmax()
        purity  = float(probs.max())
        n_toronto = int(ct.loc[cl].sum())
        rows.append({
            "sample"              : sample,
            "assigned_cluster"    : cl,
            "predicted_MG"        : pred_mg,
            "cluster_purity"      : round(purity, 3),
            "n_toronto_in_cluster": n_toronto,
            **{mg: round(float(probs.get(mg, 0.0)), 3) for mg in mg_cols},
        })

    pred_df = pd.DataFrame(rows).set_index("sample")
    pred_df.to_csv(f"{OUT_DIR}/mg4_predictions_{tag}.tsv", sep="\t")
    print(f"[SAVED] mg4_predictions_{tag}.tsv")
    print(pred_df[["assigned_cluster","predicted_MG","cluster_purity"] + mg_cols]
          .to_string())

    # ── Probability heatmap: ONT samples × MG1-4 ─────────────────────────────
    prob_mat = pred_df[mg_cols].values.astype(float)   # (12, 4)
    n_ont = len(pred_df)

    # fig = plt.figure(figsize=(10, max(4, n_ont * 0.55)), facecolor="white")
    fig = plt.figure(figsize=(10, 14), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[3, 0.12],
        wspace=0.25,
        figure=fig,
    )
    ax     = fig.add_subplot(gs[0, 0])   # heatmap
    ax_cb  = fig.add_subplot(gs[0, 1])   # colorbar
    # ax2    = fig.add_subplot(gs[0, 2])   # right-side purity bars

    # Left: probability heatmap
    im = ax.imshow(prob_mat, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(mg_cols)))
    ax.set_xticklabels(mg_cols, fontsize=FONT["axis_label"], fontfamily="monospace")
    ax.set_yticks(range(n_ont))
    ax.set_yticklabels(pred_df.index.tolist(), fontsize=FONT["ticklabel"], fontfamily="monospace")
    ax.set_xlabel("Methylation Group", fontsize=FONT["axis_label"], fontfamily="monospace")
    for i in range(n_ont):
        for j in range(len(mg_cols)):
            v = prob_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FONT["cell_annotation"], fontfamily="monospace",
                        color="white" if v > 0.6 else "#333333")

    # Colorbar in its own column
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Probability", fontsize=FONT["cbar_label_small"], fontfamily="monospace")
    cbar.ax.tick_params(labelsize=FONT["ticklabel_small"])

    # Right: predicted MG + cluster purity bars
    # imshow draws row 0 at TOP — invert ax2 to match.
    # ax2.set_xlim(0, 1.4)
    # ax2.set_ylim(n_ont - 0.5, -0.5)
    # ax2.axis("off")
    # for i, (sample, row) in enumerate(pred_df.iterrows()):
    #     mg   = row["predicted_MG"]
    #     pur  = row["cluster_purity"] if not np.isnan(row["cluster_purity"]) else 0
    #     col  = MG_BLUE_SHADES.get(mg, MG_UNKNOWN_COLOR)
    #     ax2.add_patch(mpatches.Rectangle((0, i - 0.4), pur, 0.8,
    #                                       color=col, alpha=0.95))
    #     ax2.text(pur + 0.04, i, f"{mg} ({pur:.0%})",
    #              va="center", fontsize=7, fontfamily="monospace", color="#333333")
    # ax2.set_title("Predicted MG\n(purity)", fontsize=8, fontfamily="monospace")

    fig.suptitle(
        f"MG predictions for ONT data via methylation clustering\n"
        f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}",
        fontsize=FONT["suptitle"], fontweight="bold", fontfamily="monospace")

    out = f"{OUT_DIR}/mg4_probability_heatmap_{tag}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")
    return pred_df


# =============================================================================
# Single combination
# =============================================================================
def run_combination(toronto, ont, lbl, mg4, norm, feat, probe_to_island=None):
    print(f"\n{'='*70}\n[RUN] norm={norm}, feat={feat}\n{'='*70}")

    # 1. Within-platform normalization
    if norm == "quantile":
        t_n = quantile_normalize(toronto)
        o_n = quantile_normalize(ont)
    elif norm == "mean_center":
        t_n = mean_center(toronto)
        o_n = mean_center(ont)
    else:
        raise ValueError(norm)

    # 2. Per-sample z-score
    t_z = zscore_per_sample(t_n)
    o_z = zscore_per_sample(o_n)

    # Keep probe-level versions for Toronto-only reproductions
    # (consensus clustering matches the original methodology on probes, not islands)
    t_z_probe = t_z.copy()

    # 3. Optional feature transform
    if feat == "cpg_island_rank":
        if probe_to_island is None:
            raise RuntimeError("probe_to_island required for cpg_island_rank")
        t_z = aggregate_to_cpg_island_rank(t_z, probe_to_island)
        o_z = aggregate_to_cpg_island_rank(o_z, probe_to_island)
        # Realign on shared islands
        common = t_z.index.intersection(o_z.index)
        t_z = t_z.loc[common]
        o_z = o_z.loc[common]
        print(f"[INFO] Aggregated to {len(common)} CpG islands")

    # 4. Pearson correlation: ONT × Toronto
    corr_ot = pairwise_correlation(o_z, t_z)     # (12, 121)

    # 5. MG assignment per ONT sample
    assignments = assign_mgs(corr_ot, lbl)

    # 6. Combined 133×133 correlation (for full consensus heatmap)
    all_z   = pd.concat([t_z, o_z], axis=1)
    corr_all = pairwise_correlation(all_z, all_z)

    # 7. Map 6-cluster assignments → 4-MG predictions via Toronto bridge
    #    (computed BEFORE plotting so the ONT marker strip can show MG shades)
    tag = f"{norm}_{feat}"
    pred_df = predict_mg_from_clusters(assignments, lbl, mg4, tag)
    # ont_sample → predicted_MG string (e.g. 'MG2'); used by plot functions
    ont_mg_pred = pred_df["predicted_MG"].to_dict()

    # 8. Plots
    save_projection_heatmap(
        corr_ot, assignments, lbl, tag, mg4=mg4, ont_mg_pred=ont_mg_pred)
    save_combined_heatmap(
        corr_all, lbl, ont.columns.tolist(), assignments, tag,
        mg4=mg4, ont_mg_pred=ont_mg_pred)
    save_combined_heatmap_ont_separated(
        corr_all, lbl, ont.columns.tolist(), assignments, tag,
        mg4=mg4, ont_mg_pred=ont_mg_pred)

    # 9. Toronto-only reproduction plots (no ONT samples, no MG strip needed)
    save_toronto_only_correlation_heatmap(t_z_probe, lbl, tag)
    save_toronto_only_consensus_heatmap(t_z_probe, lbl, tag)

    assignments.to_csv(f"{OUT_DIR}/mg_assignments_{tag}.tsv", sep="\t")
    print(f"[SAVED] mg_assignments_{tag}.tsv")
    return assignments


def pairwise_correlation(A, B):
    """Pearson correlation between every column of A and every column of B."""
    Av = A.values - A.values.mean(axis=0)
    Bv = B.values - B.values.mean(axis=0)
    An = Av / np.linalg.norm(Av, axis=0, keepdims=True)
    Bn = Bv / np.linalg.norm(Bv, axis=0, keepdims=True)
    return pd.DataFrame(An.T @ Bn, index=A.columns, columns=B.columns)


def assign_mgs(corr_ot, lbl):
    """For each ONT sample, mean correlation per Toronto MG; argmax = assigned MG."""
    by_mg = {}
    for mg in sorted(lbl.unique()):
        cols = lbl.index[lbl == mg]
        cols = [c for c in cols if c in corr_ot.columns]
        by_mg[int(mg)] = corr_ot[cols].mean(axis=1)
    mg_corr = pd.DataFrame(by_mg)
    top1 = mg_corr.idxmax(axis=1)
    top1_val = mg_corr.max(axis=1)
    # second-best
    second = mg_corr.apply(
        lambda r: r.drop(r.idxmax()).idxmax(), axis=1)
    second_val = mg_corr.apply(
        lambda r: r.drop(r.idxmax()).max(), axis=1)
    out = pd.DataFrame({
        "assigned_MG": top1.astype(int),
        "top1_corr"  : top1_val.round(4),
        "second_MG"  : second.astype(int),
        "second_corr": second_val.round(4),
        "margin"     : (top1_val - second_val).round(4),
    })
    out = out.sort_values("assigned_MG")
    return out


def cluster_grouped_order_and_linkage(corr, sample_to_cluster):
    """Build a leaf order that groups same-cluster samples adjacently, and a
    matching Ward linkage that reflects this order (so the rendered dendrogram
    has no branch crossings).

    Strategy:
      1. Within each cluster, do OLO on that cluster's correlation submatrix
         to get a sensible intra-cluster order and intra-cluster linkage.
      2. Stitch the per-cluster linkages together: each cluster forms a
         subtree at the bottom; clusters are merged left-to-right above them
         at a uniform high level (just above max intra-cluster height).

    Returns (order, Z). dendrogram(Z) yields leaves in `order`; intra-cluster
    branch heights are faithful Ward distances; inter-cluster merges are
    drawn at a single high level to indicate they're externally imposed.
    """
    samples = list(corr.index)
    cl_assign = pd.Series(
        {s: sample_to_cluster.get(s, 0) for s in samples}, name="cluster")
    cluster_ids = sorted(set(cl_assign.values))

    # Build linkage and order per cluster
    cluster_blocks = []   # list of (member_global_idx, Z_sub)
    order = []
    for cl in cluster_ids:
        members = [s for s in samples if cl_assign[s] == cl]
        if len(members) == 1:
            # Singleton — no linkage, just emit the leaf
            cluster_blocks.append((members, None))
            order.extend(members)
            continue
        sub = corr.loc[members, members]
        sub_dist = squareform(np.clip(1 - sub.values, 0, 2), checks=False)
        Z_sub = linkage(sub_dist, method="ward")
        try:
            Z_sub = optimal_leaf_ordering(Z_sub, sub_dist)
        except Exception:
            pass
        leaves = dendrogram(Z_sub, no_plot=True)["leaves"]
        sub_order = [members[i] for i in leaves]
        # Reindex Z_sub leaves to match sub_order (relabel by sorting permutation)
        # Easier: just store as-is; we'll reuse the per-cluster Z later.
        cluster_blocks.append((sub_order, Z_sub))
        order.extend(sub_order)

    # Stitch per-cluster linkages into one global linkage matrix in scipy format.
    # Z has shape (N-1, 4): [left, right, height, n_leaves_in_cluster].
    # Leaves are numbered 0..N-1 in the final order; internal nodes start at N.
    n_total = len(order)
    idx_of = {s: i for i, s in enumerate(order)}

    all_links = []
    cluster_roots = []   # (root_node_id, max_height_in_subtree)

    for sub_order, Z_sub in cluster_blocks:
        if Z_sub is None:
            # singleton
            cluster_roots.append((idx_of[sub_order[0]], 0.0))
            continue
        # Relabel Z_sub leaf IDs to global IDs
        # Z_sub leaves are 0..len(sub_order)-1 in the order scipy assigned them.
        # The dendrogram leaves list told us how scipy orders them; we need
        # to map scipy's leaf id i → global id of sub_order[i].
        # But Z_sub itself was built before the OLO reorder — the leaf ids in Z
        # are 0..len(members)-1 referring to the input `members` order, NOT
        # `sub_order`. So we map: scipy_leaf_id → members[scipy_leaf_id] → global idx.
        members_for_block = []
        # Reconstruct: which sample does each scipy leaf id 0..m-1 refer to?
        # It's the same as `members` in the loop above; we need to recompute.
        cl_for_block = [
            cluster_ids[i] for i, (so, _) in enumerate(cluster_blocks)
            if so is sub_order
        ][0]
        members_for_block = [s for s in samples if cl_assign[s] == cl_for_block]
        n_block = len(members_for_block)

        # Allocate internal node IDs for this block: start at n_total + offset
        offset = sum(len(all_links) for _ in [0])  # accumulated internal nodes
        # (recompute properly below)

    # Simpler robust approach: don't try to stitch. Use a single Ward linkage
    # on a synthetic distance matrix that (a) encodes real intra-cluster
    # distances and (b) makes between-cluster distances large enough that
    # within-cluster merges always happen first.
    n = len(order)
    D = np.full((n, n), 0.0, dtype=float)
    corr_ord = corr.loc[order, order].values
    base_dist = np.clip(1 - corr_ord, 0, 2)

    # Find max within-cluster distance
    max_within = 0.0
    cluster_of_ordered = [cl_assign[s] for s in order]
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_of_ordered[i] == cluster_of_ordered[j]:
                max_within = max(max_within, base_dist[i, j])

    # Inflate between-cluster distances to be larger than any within
    between_floor = max_within * 1.5 + 0.5
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_of_ordered[i] == cluster_of_ordered[j]:
                D[i, j] = D[j, i] = base_dist[i, j]
            else:
                # Make inter-cluster distance increase with cluster index gap,
                # so adjacent clusters merge before distant ones
                gap = abs(cluster_ids.index(cluster_of_ordered[i])
                          - cluster_ids.index(cluster_of_ordered[j]))
                D[i, j] = D[j, i] = between_floor + gap * 0.1

    Z = linkage(squareform(D, checks=False), method="average")
    # OLO can't change cluster grouping here (between-cluster distances are huge),
    # but it tidies up within-cluster orientation
    try:
        Z = optimal_leaf_ordering(Z, squareform(D, checks=False))
    except Exception:
        pass

    # Rebuild order from the resulting dendrogram (OLO may have flipped some siblings)
    new_leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = [order[i] for i in new_leaves]
    return order, Z


# =============================================================================
# Plots
# =============================================================================
def save_projection_heatmap(corr_ot, assignments, lbl, tag,
                            mg4=None, ont_mg_pred=None):
    """12 ONT × 121 Toronto. Two top strips: Toronto cluster + Toronto MG.
    Two side strips: ONT predicted cluster (left, with sample labels) + ONT
    predicted MG (right, blue shades)."""
    t_order = lbl.sort_values().index.tolist()
    t_order = [c for c in t_order if c in corr_ot.columns]
    o_order = assignments.index.tolist()

    mat = corr_ot.loc[o_order, t_order].values
    ont_mg_pred = ont_mg_pred or {}

    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs  = gridspec.GridSpec(
        3, 4,
        height_ratios=[0.4, 0.4, 12],
        width_ratios=[0.6, 40, 0.4, 1],
        hspace=0.04, wspace=0.04,
        figure=fig,
    )
    ax_top_cl = fig.add_subplot(gs[0, 1])   # Toronto cluster strip (top)
    ax_top_mg = fig.add_subplot(gs[1, 1])   # Toronto MG strip (just below cluster)
    ax_side   = fig.add_subplot(gs[2, 0])   # ONT cluster + labels (left)
    ax_hmap   = fig.add_subplot(gs[2, 1])
    ax_right  = fig.add_subplot(gs[2, 2])   # ONT predicted MG (right)
    ax_cbar   = fig.add_subplot(gs[2, 3])

    # Toronto cluster strip (top row)
    ax_top_cl.set_xlim(0, len(t_order))
    ax_top_cl.set_ylim(0, 1)
    ax_top_cl.axis("off")
    for i, s in enumerate(t_order):
        col = CLUSTER_COLORS.get(int(lbl.loc[s]), "#cccccc")
        ax_top_cl.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=col))

    # Toronto MG strip (second row)
    ax_top_mg.set_xlim(0, len(t_order))
    ax_top_mg.set_ylim(0, 1)
    ax_top_mg.axis("off")
    for i, s in enumerate(t_order):
        mg  = mg4.loc[s] if (mg4 is not None and s in mg4.index) else None
        col = MG_BLUE_SHADES.get(mg, MG_UNKNOWN_COLOR)
        ax_top_mg.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=col))

    # ONT left strip — predicted cluster colors + sample labels
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, len(o_order))
    ax_side.axis("off")
    for i, s in enumerate(o_order):
        col = CLUSTER_COLORS.get(int(assignments.loc[s, "assigned_MG"]), "#cccccc")
        ax_side.add_patch(mpatches.Rectangle((0, len(o_order) - 1 - i), 1, 1, color=col))
        ax_side.text(-0.2, len(o_order) - 1 - i + 0.5, s,
                     ha="right", va="center", fontsize=FONT["sample_label"], fontfamily="monospace")

    # Heatmap
    vmax = float(np.nanmax(np.abs(mat)))
    vmin = -vmax
    cmap = LinearSegmentedColormap.from_list(
        "rwb_diverge", ["#2c5aa0", "#ffffff", "#6a3d9a"])
    im = ax_hmap.pcolormesh(
        np.arange(len(t_order) + 1),
        np.arange(len(o_order) + 1),
        mat, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax_hmap.invert_yaxis()
    ax_hmap.set_xticks([])
    ax_hmap.set_yticks([])
    for sp in ax_hmap.spines.values():
        sp.set_edgecolor("#999999")

    # ONT right strip — predicted MG (blue shades)
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, len(o_order))
    ax_right.axis("off")
    for i, s in enumerate(o_order):
        mg  = ont_mg_pred.get(s, None)
        col = MG_BLUE_SHADES.get(mg, MG_UNKNOWN_COLOR)
        ax_right.add_patch(mpatches.Rectangle(
            (0, len(o_order) - 1 - i), 1, 1, color=col))

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Pearson r", fontsize=FONT["cbar_label_small"], fontfamily="monospace")

    # Projection heatmap is short — place both legends side by side below the figure
    cluster_handles = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f"{i}")
                       for i in range(1, K + 1)]
    mg_handles      = [mpatches.Patch(color=MG_BLUE_SHADES[m], label=m)
                       for m in ("MG1", "MG2", "MG3", "MG4")]
    leg1 = fig.legend(handles=cluster_handles, loc="upper left", fontsize=FONT["legend_small"],
                      framealpha=0.92, edgecolor="#cccccc",
                      title="Methylation\nCluster", title_fontsize=FONT["legend_title_small"],
                      bbox_to_anchor=(0.95, 0.95),
                      bbox_transform=fig.transFigure, ncol=1)
    fig.add_artist(leg1)
    fig.legend(handles=mg_handles, loc="upper left", fontsize=FONT["legend_small"],
               framealpha=0.92, edgecolor="#cccccc",
               title="MG", title_fontsize=FONT["legend_title_small"],
               bbox_to_anchor=(0.95, 0.45),
               bbox_transform=fig.transFigure, ncol=1)

    fig.subplots_adjust(top=0.88)
    fig.suptitle(
        f"ONT samples projected onto Nassiri et al. reference\n"
        f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}",
        fontsize=FONT["suptitle_small"], fontweight="bold", fontfamily="monospace", y=0.97)

    out = f"{OUT_DIR}/projection_heatmap_{tag}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def _cluster_for_row(s, lbl, ont_set, assignments):
    """Return cluster color for a given sample, looking it up in lbl
    (Toronto) or assignments (ONT)."""
    if s in ont_set:
        if assignments is not None and s in assignments.index:
            return CLUSTER_COLORS.get(int(assignments.loc[s, "assigned_MG"]),
                                       "#cccccc")
        return "#dddddd"
    if s in lbl.index:
        return CLUSTER_COLORS.get(int(lbl.loc[s]), "#cccccc")
    return "#cccccc"


def _mg_color_for_row(s, mg4, ont_set, ont_mg_pred):
    """Return MG color (blue gradient) for a sample. For Toronto: known MG from
    mg_sample_labels.tsv. For ONT: predicted MG."""
    if s in ont_set:
        mg = (ont_mg_pred or {}).get(s, None)
    elif mg4 is not None and s in mg4.index:
        mg = mg4.loc[s]
    else:
        mg = None
    return MG_BLUE_SHADES.get(mg, MG_UNKNOWN_COLOR)


def _draw_ont_labels_x(ax_hmap, order, ont_cols, leaf_pos, fontsize=FONT["sample_label"]):
    """Add x-axis tick labels for ONT samples only, staggered vertically
    (alternate pad) so adjacent labels don't overlap. Labels sit below the axis."""
    ont_set = set(ont_cols)
    ont_items = [(xpos, s) for s, xpos in zip(order, leaf_pos) if s in ont_set]
    if not ont_items:
        return
    ax_hmap.set_xticks([x for x, _ in ont_items])
    ax_hmap.set_xticklabels([], fontsize=fontsize)
    # Manual stagger: even-index labels just below axis, odd deeper below
    for idx, (xpos, s) in enumerate(ont_items):
        pad = -3 if idx % 2 == 0 else -28
        ax_hmap.annotate(
            s, xy=(xpos, 0), xycoords=("data", "axes fraction"),
            xytext=(0, pad), textcoords="offset points",
            ha="center", va="top", fontsize=fontsize,
            fontfamily="monospace", rotation=90,
            annotation_clip=False)
    ax_hmap.tick_params(axis="x", which="both", length=2, pad=0)


def _draw_ont_labels_y(ax_hmap, order, ont_cols, leaf_pos, fontsize=FONT["sample_label"]):
    """Add y-axis tick labels for ONT samples only, staggered horizontally
    (alternate pad) so adjacent labels don't overlap."""
    ont_set = set(ont_cols)
    ont_items = [(ypos, s) for s, ypos in zip(order, leaf_pos) if s in ont_set]
    if not ont_items:
        return
    ax_hmap.set_yticks([y for y, _ in ont_items])
    ax_hmap.set_yticklabels([], fontsize=fontsize)
    # Manual stagger: even-index labels at pad=-4, odd at pad=-18 (further left)
    for idx, (ypos, s) in enumerate(ont_items):
        pad = -3 if idx % 2 == 0 else -28
        ax_hmap.annotate(
            s, xy=(0, ypos), xycoords=("axes fraction", "data"),
            xytext=(pad, 0), textcoords="offset points",
            ha="right", va="center", fontsize=fontsize,
            fontfamily="monospace", annotation_clip=False)
    ax_hmap.tick_params(axis="y", which="both", length=2, pad=0)


def _draw_ont_red_markers(ax_hmap, order, ont_cols, leaf_pos, x_hi):
    """Draw red square markers just outside the heatmap spines — touching the
    border but not overlapping cells.
    - x-axis markers sit at the BOTTOM edge (near patient name labels)
    - y-axis markers sit at the LEFT edge"""
    from matplotlib.transforms import offset_copy
    ont_set = set(ont_cols)

    # Bottom edge: x in data coords, y just below bottom spine (small pixel nudge down)
    x_trans_bot = offset_copy(
        blended_transform_factory(ax_hmap.transData, ax_hmap.transAxes),
        fig=ax_hmap.figure, x=0, y=-3, units="points")
    # Left edge: y in data coords, x just left of left spine (smaller nudge)
    y_trans_left = offset_copy(
        blended_transform_factory(ax_hmap.transAxes, ax_hmap.transData),
        fig=ax_hmap.figure, x=-1, y=0, units="points")

    for s, xpos in zip(order, leaf_pos):
        if s in ont_set:
            ax_hmap.plot(xpos, 0.0, "rs", markersize=4, clip_on=False,
                         zorder=6, transform=x_trans_bot)
            ax_hmap.plot(0.0, xpos, "rs", markersize=4, clip_on=False,
                         zorder=6, transform=y_trans_left)

def _add_legends_top_right(fig, is_mixed_toronto_ont=True,
                            cluster_x=1.01, cluster_y=0.70,
                            mg_y=0.45):
    """Add two stacked vertical legends top-right, outside the heatmap.
    Methylation Cluster legend first, MG legend just below it."""
    cluster_handles = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f"{i}")
                       for i in range(1, K + 1)]
    mg_handles      = [mpatches.Patch(color=MG_BLUE_SHADES[m], label=m)
                       for m in ("MG1", "MG2", "MG3", "MG4")]
    mg_title = "MG"
    leg1 = fig.legend(handles=cluster_handles, loc="upper left", fontsize=FONT["legend"],
                      framealpha=0.92, edgecolor="#cccccc",
                      title="Methylation\nCluster", title_fontsize=FONT["legend_title"],
                      bbox_to_anchor=(cluster_x, cluster_y),
                      bbox_transform=fig.transFigure)
    fig.add_artist(leg1)
    fig.legend(handles=mg_handles, loc="upper left", fontsize=FONT["legend"],
               framealpha=0.92, edgecolor="#cccccc",
               title=mg_title, title_fontsize=FONT["legend_title"],
               bbox_to_anchor=(cluster_x, mg_y),
               bbox_transform=fig.transFigure)


def _set_title(fig, text, fontsize=FONT["suptitle_medium"]):
    """Set figure title closer to the dendrogram with larger font.
    Tightens top margin so the title sits just above the dendrogram."""
    fig.subplots_adjust(top=0.93)
    fig.suptitle(text, fontsize=fontsize, fontweight="bold",
                 fontfamily="monospace", y=0.97)


def save_combined_heatmap(corr_all, lbl, ont_cols, assignments, tag,
                          mg4=None, ont_mg_pred=None):
    """133×133 with dendrogram, top cluster strip, MG strip (all 133 samples
    colored by MG: Toronto by known MG, ONT by predicted MG), right-side
    cluster strip, and ONT labels."""
    n = len(corr_all)
    ont_mg_pred = ont_mg_pred or {}
    # Build cluster map: Toronto from lbl, ONT from predicted assignments
    sample_to_cluster = {}
    for s in corr_all.index:
        if s in set(ont_cols):
            if s in assignments.index:
                sample_to_cluster[s] = int(assignments.loc[s, "assigned_MG"])
            else:
                sample_to_cluster[s] = 0
        elif s in lbl.index:
            sample_to_cluster[s] = int(lbl.loc[s])
        else:
            sample_to_cluster[s] = 0
    order, Z = cluster_grouped_order_and_linkage(corr_all, sample_to_cluster)
    mat = corr_all.loc[order, order].values

    fig = plt.figure(figsize=(14, 14), facecolor="white")
    # 4 rows (dend, top cluster strip, MG strip, heatmap)
    # 3 cols (heatmap area, right cluster strip, colorbar)
    gs  = gridspec.GridSpec(
        4, 3,
        height_ratios=[1.5, 0.12, 0.12, 10],
        width_ratios=[40, 0.5, 1],
        hspace=0.01, wspace=0.02,
        figure=fig,
    )
    ax_dend  = fig.add_subplot(gs[0, 0])
    ax_mg_cl = fig.add_subplot(gs[1, 0], sharex=ax_dend)
    ax_mg    = fig.add_subplot(gs[2, 0], sharex=ax_dend)
    ax_hmap  = fig.add_subplot(gs[3, 0], sharex=ax_dend)
    ax_right = fig.add_subplot(gs[3, 1], sharey=ax_hmap)
    ax_cbar  = fig.add_subplot(gs[3, 2])

    # Dendrogram with warm color gradient
    heights      = Z[:, 2]
    h_min, h_max = heights.min(), heights.max()
    warm = ["#f5e642","#e8c84a","#d4a843","#c07a2e","#a04e1a","#6b2a0a"]
    def branch_color(k):
        n_leaves = Z.shape[0] + 1
        if k < n_leaves:
            return warm[0]
        h    = Z[k - n_leaves, 2]
        norm_h = (h - h_min) / (h_max - h_min) if h_max > h_min else 0
        return warm[min(int(norm_h * (len(warm) - 1)), len(warm) - 1)]

    dendrogram(Z, ax=ax_dend, no_labels=True,
               color_threshold=0, link_color_func=branch_color)
    ax_dend.axis("off")
    x_lo, x_hi = 0, n * 10
    ax_dend.set_xlim(x_lo, x_hi)

    leaf_pos = np.array([5 + i * 10 for i in range(n)])
    ont_set = set(ont_cols)

    # Top cluster strip
    ax_mg_cl.set_facecolor("white")
    ax_mg_cl.set_ylim(0, 1)
    ax_mg_cl.axis("off")
    for s, xpos in zip(order, leaf_pos):
        col = _cluster_for_row(s, lbl, ont_set, assignments)
        ax_mg_cl.add_patch(mpatches.Rectangle((xpos - 5, 0), 10, 1, color=col, zorder=3))

    # MG strip — every sample colored by MG (Toronto known, ONT predicted)
    ax_mg.set_facecolor("white")
    ax_mg.set_ylim(0, 1)
    ax_mg.axis("off")
    for s, xpos in zip(order, leaf_pos):
        col = _mg_color_for_row(s, mg4, ont_set, ont_mg_pred)
        ax_mg.add_patch(mpatches.Rectangle(
            (xpos - 5, 0), 10, 1, color=col, zorder=3))

    # Heatmap
    ax_hmap.set_facecolor("white")
    x_edges = np.arange(n + 1) * 10
    y_edges = np.arange(n + 1) * 10
    im = ax_hmap.pcolormesh(
        x_edges, y_edges, mat,
        cmap=PURPLE_CMAP, vmin=0, vmax=1, rasterized=True)
    ax_hmap.set_xlim(x_lo, x_hi)
    ax_hmap.set_ylim(x_hi, x_lo)  # invert: row 0 at top
    for sp in ax_hmap.spines.values():
        sp.set_edgecolor("#999999")
    _draw_ont_labels_x(ax_hmap, order, ont_cols, leaf_pos)
    _draw_ont_labels_y(ax_hmap, order, ont_cols, leaf_pos)
    _draw_ont_red_markers(ax_hmap, order, ont_cols, leaf_pos, x_hi)

    # Right-side cluster strip (mirrors top strip but vertical)
    ax_right.set_facecolor("white")
    ax_right.set_xlim(0, 1)
    ax_right.axis("off")
    for s, ypos in zip(order, leaf_pos):
        col = _cluster_for_row(s, lbl, ont_set, assignments)
        ax_right.add_patch(mpatches.Rectangle((0, ypos - 5), 1, 10, color=col, zorder=3))

    # Colorbar
    cbar_pos = ax_cbar.get_position()
    new_height = cbar_pos.height * 0.5
    new_y      = cbar_pos.y0 + (cbar_pos.height - new_height) / 2
    ax_cbar.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Pearson r", fontsize=FONT["cbar_label_large"], fontfamily="monospace")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    _add_legends_top_right(fig, is_mixed_toronto_ont=True, cluster_x=0.95, cluster_y=0.70, mg_y=0.48)
    _set_title(fig,
        f"Combined 135-sample consensus  ·  Nassiri et al. (121) + ONT (14)\n"
               f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}",
    fontsize=FONT["suptitle"] )

    out = f"{OUT_DIR}/combined_consensus_{tag}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def save_combined_heatmap_ont_separated(corr_all, lbl, ont_cols, assignments, tag,
                                          mg4=None, ont_mg_pred=None):
    """133×133 layout with Toronto (121) clustered on its own at top-left and
    ONT (12) clustered on its own at bottom-right. ONT samples labeled by name
    on heatmap axes. Top and right cluster strips show known (Toronto) and
    predicted (ONT) clusters. Full MG strip (Toronto known, ONT predicted)
    colors all 133 samples."""
    ont_mg_pred = ont_mg_pred or {}
    toronto_cols = [c for c in corr_all.columns if c not in set(ont_cols)]
    n_t = len(toronto_cols)
    n_o = len(ont_cols)
    n   = n_t + n_o

    corr_t = corr_all.loc[toronto_cols, toronto_cols]
    # Cluster-grouped order for Toronto block (known labels)
    t_to_cluster = {s: int(lbl.loc[s]) for s in toronto_cols if s in lbl.index}
    t_order, Z_t = cluster_grouped_order_and_linkage(corr_t, t_to_cluster)

    corr_o = corr_all.loc[ont_cols, ont_cols]
    if n_o > 1:
        # Cluster-grouped order for ONT block (predicted labels)
        o_to_cluster = {s: int(assignments.loc[s, "assigned_MG"])
                        for s in ont_cols if s in assignments.index}
        o_order, Z_o = cluster_grouped_order_and_linkage(corr_o, o_to_cluster)
    else:
        o_order = list(ont_cols)
        Z_o = None

    order = t_order + o_order
    mat   = corr_all.loc[order, order].values

    fig = plt.figure(figsize=(14, 14), facecolor="white")
    gs  = gridspec.GridSpec(
        4, 3,
        height_ratios=[1.5, 0.12, 0.12, 10],
        width_ratios=[40, 0.5, 1],
        hspace=0.01, wspace=0.02,
        figure=fig,
    )
    ax_dend  = fig.add_subplot(gs[0, 0])
    ax_mg    = fig.add_subplot(gs[1, 0])
    ax_ont   = fig.add_subplot(gs[2, 0])
    ax_hmap  = fig.add_subplot(gs[3, 0])
    ax_right = fig.add_subplot(gs[3, 1])
    ax_cbar  = fig.add_subplot(gs[3, 2])

    x_lo, x_hi = 0, n * 10
    t_x_hi = n_t * 10
    leaf_pos = np.array([5 + i * 10 for i in range(n)])
    ont_set = set(ont_cols)

    # Toronto-only dendrogram
    heights      = Z_t[:, 2]
    h_min, h_max = heights.min(), heights.max()
    warm = ["#f5e642","#e8c84a","#d4a843","#c07a2e","#a04e1a","#6b2a0a"]
    def branch_color(k):
        n_leaves = Z_t.shape[0] + 1
        if k < n_leaves:
            return warm[0]
        h    = Z_t[k - n_leaves, 2]
        norm_h = (h - h_min) / (h_max - h_min) if h_max > h_min else 0
        return warm[min(int(norm_h * (len(warm) - 1)), len(warm) - 1)]

    dendrogram(Z_t, ax=ax_dend, no_labels=True,
               color_threshold=0, link_color_func=branch_color)
    ax_dend.set_xlim(x_lo, x_hi)
    ax_dend.axis("off")

    # Top cluster strip — Toronto known + ONT predicted
    ax_mg.set_facecolor("white")
    ax_mg.set_xlim(x_lo, x_hi)
    ax_mg.set_ylim(0, 1)
    ax_mg.axis("off")
    for s, xpos in zip(order, leaf_pos):
        col = _cluster_for_row(s, lbl, ont_set, assignments)
        ax_mg.add_patch(mpatches.Rectangle((xpos - 5, 0), 10, 1, color=col, zorder=3))

    # MG strip — every sample colored by MG (Toronto known, ONT predicted)
    ax_ont.set_facecolor("white")
    ax_ont.set_xlim(x_lo, x_hi)
    ax_ont.set_ylim(0, 1)
    ax_ont.axis("off")
    for s, xpos in zip(order, leaf_pos):
        col = _mg_color_for_row(s, mg4, ont_set, ont_mg_pred)
        ax_ont.add_patch(mpatches.Rectangle(
            (xpos - 5, 0), 10, 1, color=col, zorder=3))

    # Heatmap
    ax_hmap.set_facecolor("white")
    x_edges = np.arange(n + 1) * 10
    y_edges = np.arange(n + 1) * 10
    im = ax_hmap.pcolormesh(
        x_edges, y_edges, mat,
        cmap=PURPLE_CMAP, vmin=0, vmax=1, rasterized=True)
    ax_hmap.set_xlim(x_lo, x_hi)
    ax_hmap.set_ylim(x_hi, x_lo)
    for sp in ax_hmap.spines.values():
        sp.set_edgecolor("#999999")
    _draw_ont_labels_x(ax_hmap, order, ont_cols, leaf_pos)
    _draw_ont_labels_y(ax_hmap, order, ont_cols, leaf_pos)
    _draw_ont_red_markers(ax_hmap, order, ont_cols, leaf_pos, x_hi)

    # Dividers
    split = t_x_hi
    ax_hmap.axvline(split, color="#333333", lw=0.8, alpha=0.7)
    ax_hmap.axhline(split, color="#333333", lw=0.8, alpha=0.7)

    # Right-side cluster strip — Toronto known + ONT predicted
    ax_right.set_facecolor("white")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(x_hi, x_lo)
    ax_right.axis("off")
    for s, ypos in zip(order, leaf_pos):
        col = _cluster_for_row(s, lbl, ont_set, assignments)
        ax_right.add_patch(mpatches.Rectangle((0, ypos - 5), 1, 10, color=col, zorder=3))

    cbar_pos = ax_cbar.get_position()
    new_height = cbar_pos.height * 0.5
    new_y      = cbar_pos.y0 + (cbar_pos.height - new_height) / 2
    ax_cbar.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Pearson r", fontsize=FONT["cbar_label"], fontfamily="monospace")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    _add_legends_top_right(fig, is_mixed_toronto_ont=True, cluster_x=0.95, cluster_y=0.70, mg_y=0.48)
    _set_title(fig,
        f"Combined heatmap (ONT appended)  ·  Nassiri et al. (121) + ONT (14) at bottom-right\n"
        f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}")

    out = f"{OUT_DIR}/combined_consensus_ont_separated_{tag}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out}")


def save_toronto_only_correlation_heatmap(toronto_z, lbl, tag):
    """121×121 Pearson correlation heatmap of Toronto-only normalized data.
    Same plotting style as the original ref consensus heatmap, but the
    cell values are correlations (not co-clustering probabilities)."""
    print(f"[INFO] Toronto-only correlation heatmap ({tag})")
    corr = pairwise_correlation(toronto_z, toronto_z)
    n = len(corr)

    sample_to_cluster = {s: int(lbl.loc[s]) for s in corr.index if s in lbl.index}
    order, Z = cluster_grouped_order_and_linkage(corr, sample_to_cluster)
    mat = corr.loc[order, order].values

    _render_toronto_only_heatmap(
        mat=mat, order=order, Z=Z, lbl=lbl,
        cbar_label="Pearson correlation",
        title=(f"Nassiri et al. only (n=121) — Pearson correlation\n"
               f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}"),
        outpath=f"{OUT_DIR}/toronto_only_correlation_{tag}.pdf",
    )


def save_toronto_only_consensus_heatmap(toronto_z, lbl, tag,
                                         n_iter=1000, subsample_frac=0.80, k=6):
    """Full consensus clustering on Toronto-only normalized data.
    Plots the 121×121 consensus matrix (co-clustering probability)."""
    print(f"[INFO] Toronto-only consensus clustering ({tag}, {n_iter} iter)")
    samples = toronto_z.columns.tolist()
    n       = len(samples)
    # Each iteration uses samples × features matrix; mean/quantile-normalized values
    mat_sf  = toronto_z.values.T   # (n_samples, n_features)

    co_occur = np.zeros((n, n), dtype=np.float32)
    selected = np.zeros((n, n), dtype=np.float32)

    rng = np.random.default_rng(seed=0)
    for it in range(n_iter):
        if (it + 1) % 100 == 0:
            print(f"  iter {it + 1}/{n_iter}")
        n_sub   = int(n * subsample_frac)
        sub_idx = np.sort(rng.choice(n, n_sub, replace=False))
        # selected counter
        sel_outer = np.ix_(sub_idx, sub_idx)
        selected[sel_outer] += 1
        # cluster the subsample
        sub_mat = mat_sf[sub_idx, :]
        dist    = np.clip(1 - np.corrcoef(sub_mat), 0, 2)
        np.fill_diagonal(dist, 0)
        Z_it    = linkage(squareform(dist, checks=False), method="ward")
        labels  = fcluster(Z_it, k, criterion="maxclust")
        # co-occurrence: pairs in the same cluster
        for cl in np.unique(labels):
            members = sub_idx[labels == cl]
            mem_outer = np.ix_(members, members)
            co_occur[mem_outer] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(selected > 0, co_occur / selected, 0.0)
    np.fill_diagonal(consensus, 1.0)
    consensus_df = pd.DataFrame(consensus, index=samples, columns=samples)
    consensus_df.to_pickle(f"{OUT_DIR}/consensus_matrix_toronto_only_{tag}.pkl")

    # Final ordering: cluster-grouped to match the pre-computed labels
    sample_to_cluster = {s: int(lbl.loc[s]) for s in samples if s in lbl.index}
    order, Z_f = cluster_grouped_order_and_linkage(consensus_df, sample_to_cluster)
    mat = consensus_df.loc[order, order].values

    _render_toronto_only_heatmap(
        mat=mat, order=order, Z=Z_f, lbl=lbl,
        cbar_label="Co-clustering probability",
        title=(f"Nassiri et al. only (n=121) — consensus clustering\n"
               f"k={k}, {n_iter} iter, {subsample_frac:.0%} subsampling  ·  "
               f"norm={tag.split('_')[0]}, features={'_'.join(tag.split('_')[1:])}"),
        outpath=f"{OUT_DIR}/toronto_only_consensus_{tag}.pdf",
    )


def _render_toronto_only_heatmap(mat, order, Z, lbl, cbar_label, title, outpath):
    """Shared rendering for the two Toronto-only heatmaps.
    Matches the layout of the original toronto_ref_consensus_heatmap."""
    n = len(order)

    fig = plt.figure(figsize=(12, 13), facecolor="white")
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.5, 0.15, 10],
        width_ratios=[40, 1],
        hspace=0.01, wspace=0.02,
        figure=fig,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_cb   = fig.add_subplot(gs[1, 0], sharex=ax_dend)
    ax_hmap = fig.add_subplot(gs[2, 0], sharex=ax_dend)
    ax_cbar = fig.add_subplot(gs[2, 1])

    # Dendrogram
    heights      = Z[:, 2]
    h_min, h_max = heights.min(), heights.max()
    warm = ["#f5e642","#e8c84a","#d4a843","#c07a2e","#a04e1a","#6b2a0a"]
    def branch_color(k):
        n_leaves = Z.shape[0] + 1
        if k < n_leaves:
            return warm[0]
        h    = Z[k - n_leaves, 2]
        norm_h = (h - h_min) / (h_max - h_min) if h_max > h_min else 0
        return warm[min(int(norm_h * (len(warm) - 1)), len(warm) - 1)]

    dendrogram(Z, ax=ax_dend, no_labels=True,
               color_threshold=0, link_color_func=branch_color)
    ax_dend.axis("off")
    x_lo, x_hi = 0, n * 10
    ax_dend.set_xlim(x_lo, x_hi)

    leaf_pos = np.array([5 + i * 10 for i in range(n)])

    # Cluster strip (from pre-computed consensus_labels_k6.pkl)
    ax_cb.set_facecolor("white")
    ax_cb.set_ylim(0, 1)
    ax_cb.axis("off")
    for s, xpos in zip(order, leaf_pos):
        col = CLUSTER_COLORS.get(int(lbl.loc[s]), "#cccccc") if s in lbl.index else "#cccccc"
        ax_cb.add_patch(mpatches.Rectangle((xpos - 5, 0), 10, 1, color=col, zorder=3))

    # Heatmap
    ax_hmap.set_facecolor("white")
    x_edges = np.arange(n + 1) * 10
    y_edges = np.arange(n + 1) * 10
    im = ax_hmap.pcolormesh(x_edges, y_edges, mat,
                            cmap=PURPLE_CMAP, vmin=0, vmax=1, rasterized=True)
    ax_hmap.set_xlim(x_lo, x_hi)
    ax_hmap.set_ylim(x_hi, x_lo)
    ax_hmap.set_xticks([])
    ax_hmap.set_yticks([])
    for sp in ax_hmap.spines.values():
        sp.set_edgecolor("#999999")

    # Colorbar
    cbar_pos = ax_cbar.get_position()
    new_height = cbar_pos.height * 0.5
    new_y      = cbar_pos.y0 + (cbar_pos.height - new_height) / 2
    ax_cbar.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=FONT["cbar_label"], fontfamily="monospace")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    handles = [mpatches.Patch(color=CLUSTER_COLORS[i], label=f"{i}")
               for i in range(1, K + 1)]
    fig.legend(handles=handles, loc="upper left", fontsize=FONT["legend_small"],
               framealpha=0.92, edgecolor="#cccccc",
               title="Methylation\nCluster", title_fontsize=FONT["legend_title_small"],
               bbox_to_anchor=(1.01, 0.92),
               bbox_transform=fig.transFigure)

    _set_title(fig, title)

    plt.savefig(outpath, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {outpath}")


# =============================================================================
# Main
# =============================================================================
def main():
    toronto, ont, lbl, mg4 = load_inputs()

    # Probe → island map (only needed if any combination uses cpg_island_rank)
    probe_to_island = load_probe_to_island_map(toronto.index)

    summary = {}
    for norm in ["quantile", "mean_center"]:
        for feat in ["probe", "cpg_island_rank"]:
            tag = f"{norm}_{feat}"
            summary[tag] = run_combination(
                toronto, ont, lbl, mg4, norm, feat,
                probe_to_island=probe_to_island)

    # Concordance table: are cluster assignments consistent across 4 settings?
    print(f"\n{'='*70}\nCluster assignment concordance across configurations\n{'='*70}")
    concord = pd.DataFrame({tag: s["assigned_MG"] for tag, s in summary.items()})
    print(concord.to_string())
    concord.to_csv(f"{OUT_DIR}/mg_assignments_concordance.tsv", sep="\t")
    print(f"\n[SAVED] {OUT_DIR}/mg_assignments_concordance.tsv")


if __name__ == "__main__":
    main()
