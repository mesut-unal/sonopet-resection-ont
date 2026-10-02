"""
promoter_methylation_heatmap.py
================================
Two cohort-agnostic views of promoter CpG methylation. No patient / condition
mapping — samples are whatever the cohort has finished.

  1. Binned density of promoter methylation for ANY pair of samples
     (replaces the old fixed R-vs-S patient plot).
  2. Marker-gene heatmap: genes x samples mean methylation, all samples.

Usage:
  # every pairwise density plot in the cohort
  python promoter_methylation_heatmap.py --cohort ONTWGS10 --pairs all

  # one specific pair
  python promoter_methylation_heatmap.py --cohort ONTWGS10 \
      --pair ONTWGS10-1-XX-NP01 ONTWGS10-2-YY-NP01

  # marker gene heatmap over all samples (coords come from gene_coords.py)
  python promoter_methylation_heatmap.py --cohort ONTWGS12 --marker-heatmap \
      --genes TBXT MGMT S100B SCGN ACADL MCM2

  # narrow TSS window instead of the full gene body
  python promoter_methylation_heatmap.py --cohort ONTWGS12 --marker-heatmap \
      --genes TBXT --promoter-window 1000

Requirements:
  conda activate methylation_plots
  conda install -c bioconda bedtools    (pairwise density only)
"""

import os
import sys
import gzip
import itertools
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_coords import GENE_COORDS, GENE_STRAND
from sample_utils import (out_base, list_samples, bedmethyl_path,
                          short_label, tabix_region)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

GENOME       = "hg38"
MIN_COVERAGE = 5
N_BINS       = 100

PLOT_CFG = dict(
    fontsize_title  = 14,
    fontsize_label  = 14,
    fontsize_tick   = 14,
    fontsize_cbar   = 14,
    fontsize_annot  = 14,
    fontsize_cpg    = 14,
    fontfamily      = "monospace",
)

DEFAULT_MARKER_GENES = ["S100B", "SCGN", "ACADL", "MCM2"]


def _gene_tag(genes, maxlen=60):
    """Compact filename tag for a gene set (hash if too long)."""
    tag = "-".join(g.replace("/", "-") for g in genes)
    if len(tag) > maxlen:
        import hashlib                            # noqa: PLC0415
        h = hashlib.md5("_".join(genes).encode()).hexdigest()[:8]
        tag = f"{len(genes)}genes_{h}"
    return tag

# Optional biological annotation shown under the gene label (not a sample map)
GENE_ANNOT = {
    "S100B": "MG1", "SCGN": "MG2", "ACADL": "MG3", "MCM2": "MG4",
    "TBXT": "chordoma", "FN1": "chordoma", "SOX9": "chordoma", "MGMT": "GBM",
    "CDKN2A/B": "tumor-supp", "CDKN2A": "tumor-supp", "CDKN2B": "tumor-supp",
    "EGFR": "oncogene",
}
ANNOT_COLORS = {
    "MG1": "#e74c3c", "MG2": "#3498db", "MG3": "#2ecc71", "MG4": "#9b59b6",
    "chordoma": "#d6604d", "GBM": "#2166ac",
    "tumor-supp": "#4393c3", "oncogene": "#b2182b",
}

# ─── PAIRWISE PROMOTER DENSITY ────────────────────────────────────────────────

def intersect_promoters(bedmethyl_path_, label, promoter_bed, cache_dir):
    """
    Intersect a bedMethyl with promoter regions.
    Returns Series: promoter_name -> mean % methylation.
    Cached per sample so it is reused across every pair.
    """
    out_isect = f"{cache_dir}/{label}_promoter_intersect.bed"
    if not os.path.exists(out_isect):
        print(f"  [bedtools] intersecting {label}...")
        cmd = ["bedtools", "intersect", "-a", promoter_bed,
               "-b", bedmethyl_path_, "-wa", "-wb"]
        with open(out_isect, "w") as fh:
            subprocess.run(cmd, stdout=fh, check=True)
    else:
        print(f"  [cache] {label}")

    # promoter cols 0-5, then bedMethyl: +9 = coverage (col 10), +10 = freq (16)
    rows = []
    with open(out_isect) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 17:
                continue
            try:
                coverage = int(f[10])
                freq     = float(f[16])
            except (ValueError, IndexError):
                continue
            if coverage < MIN_COVERAGE:
                continue
            rows.append({"promoter": f[3], "freq": freq})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float, name=label)
    return df.groupby("promoter")["freq"].mean().rename(label)


def plot_pair_density(cohort, sample_a, sample_b, out_dir, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    la, lb = short_label(sample_a, cohort), short_label(sample_b, cohort)
    out_pdf = f"{out_dir}/promoter_density_{la}_vs_{lb}.pdf"
    if os.path.exists(out_pdf) and not overwrite:
        print(f"  [SKIP] exists: {out_pdf}")
        return

    promoter_bed = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "hg38_promoters_tss2kb.bed")
    if not os.path.exists(promoter_bed):
        raise FileNotFoundError(f"Promoter BED not found: {promoter_bed}")

    cache_dir = f"{out_dir}/cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[INFO] {la} vs {lb}")
    sa = intersect_promoters(bedmethyl_path(cohort, sample_a), la,
                             promoter_bed, cache_dir)
    sb = intersect_promoters(bedmethyl_path(cohort, sample_b), lb,
                             promoter_bed, cache_dir)

    mat = pd.concat([sa, sb], axis=1).dropna()
    print(f"  shared promoters (cov>={MIN_COVERAGE}x): {len(mat):,}")
    if len(mat) < 10:
        print("  [WARN] too few shared promoters, skipping")
        return

    x = mat[la].values
    y = mat[lb].values
    r_val, _ = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor="white")
    ax.set_facecolor("white")

    h, _, _ = np.histogram2d(x, y, bins=N_BINS, range=[[0, 100], [0, 100]])
    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")
    h_log = np.log2(np.where(h == 0, np.nan, h)).T

    im = ax.imshow(h_log, origin="lower", extent=[0, 100, 0, 100],
                   aspect="auto", cmap=cmap, vmin=0.01,
                   interpolation="nearest")
    ax.plot([0, 100], [0, 100], color="#333333", linewidth=0.8,
            linestyle="--", alpha=0.4)

    ax.set_xlabel(f"% Methylation  [{la}]", color="#333333",
                  fontsize=PLOT_CFG["fontsize_label"],
                  fontfamily=PLOT_CFG["fontfamily"])
    ax.set_ylabel(f"% Methylation  [{lb}]", color="#333333",
                  fontsize=PLOT_CFG["fontsize_label"],
                  fontfamily=PLOT_CFG["fontfamily"])
    ax.set_title(f"Promoter CpG methylation  ·  {cohort}",
                 color="#111111", fontsize=PLOT_CFG["fontsize_title"],
                 fontweight="bold", fontfamily=PLOT_CFG["fontfamily"], pad=12)
    ax.tick_params(colors="#333333", labelsize=PLOT_CFG["fontsize_tick"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2(promoters per bin)", color="#333333",
                   fontsize=PLOT_CFG["fontsize_cbar"],
                   fontfamily=PLOT_CFG["fontfamily"])
    cbar.ax.tick_params(labelsize=PLOT_CFG["fontsize_tick"])
    cbar.outline.set_edgecolor("#cccccc")

    ax.text(0.04, 0.95, f"r = {r_val:.3f}\nn = {len(mat):,} promoters",
            transform=ax.transAxes, color="#222222",
            fontsize=PLOT_CFG["fontsize_annot"],
            fontfamily=PLOT_CFG["fontfamily"], va="top")

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [SAVED] {out_pdf}")
    return {"sample_a": sample_a, "sample_b": sample_b,
            "n_promoters": len(mat), "pearson_r": r_val}


# ─── MARKER GENE HEATMAP ──────────────────────────────────────────────────────

def gene_window(gene, promoter_window=None):
    """
    (chrom, start, end) for a gene. With --promoter-window w, take w bp from
    the TSS end of the gene, using GENE_STRAND when known.
    """
    c = GENE_COORDS[gene][GENOME]
    chrom = c["chr"] if str(c["chr"]).startswith("chr") else f"chr{c['chr']}"
    start, end = c["start"], c["end"]
    if not promoter_window:
        return chrom, start, end

    strand = c.get("strand") or GENE_STRAND.get(gene)
    if strand is None:
        print(f"  [WARN] strand unknown for {gene}; assuming '+' "
              f"(add it to GENE_STRAND in gene_coords.py)")
        strand = "+"
    if strand == "-":
        return chrom, max(0, end - promoter_window), end
    return chrom, start, start + promoter_window


def mean_methylation_region(bed_path, chrom, start, end, min_cov=MIN_COVERAGE):
    """Mean % methylation over CpG sites in a region (tabix when available)."""
    def _mean(lines):
        vals = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 11 or f[0] != chrom:
                continue
            try:
                pos  = int(f[1])
                cov  = int(f[9])
                freq = float(f[10])
            except ValueError:
                continue
            if pos < start or pos > end or cov < min_cov:
                continue
            vals.append(freq)
        return (np.mean(vals), len(vals)) if vals else (np.nan, 0)

    lines = tabix_region(bed_path, chrom, start, end)
    if lines is not None:
        return _mean(lines)
    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        return _mean(fh)


def plot_marker_heatmap(cohort, samples, genes, out_dir,
                        promoter_window=None, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    gtag = _gene_tag(genes)
    win = f"tss{promoter_window}" if promoter_window else "fullgene"
    suffix = f"_{gtag}_{win}"
    out_pdf = f"{out_dir}/marker_gene_heatmap{suffix}.pdf"
    if os.path.exists(out_pdf) and not overwrite:
        print(f"[SKIP] exists: {out_pdf}")
        return

    labels = [short_label(s, cohort) for s in samples]
    mat = pd.DataFrame(index=labels, columns=genes, dtype=float)

    for gene in genes:
        chrom, start, end = gene_window(gene, promoter_window)
        print(f"[INFO] {gene}  {chrom}:{start:,}-{end:,}")
        for sid, lbl in zip(samples, labels):
            val, n = mean_methylation_region(
                bedmethyl_path(cohort, sid), chrom, start, end)
            mat.loc[lbl, gene] = val
            print(f"    {lbl}: " +
                  (f"{val:.1f}%  (n={n} CpG)" if not np.isnan(val) else "NA"))

    vals = mat.values.astype(float)
    if np.all(np.isnan(vals)):
        print("[ERROR] no data in any region"); return
    vmin = max(0,   np.floor(np.nanmin(vals) / 5) * 5)
    vmax = min(100, np.ceil(np.nanmax(vals) / 5) * 5)

    fig, ax = plt.subplots(
        figsize=(1.6 * len(genes) + 3, 0.55 * len(labels) + 3),
        facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(vals, aspect="auto", cmap="RdYlBu_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    mid = (vmin + vmax) / 2
    for i in range(len(labels)):
        for j in range(len(genes)):
            v = vals[i, j]
            txt, col = ("NA", "#aaaaaa") if np.isnan(v) else \
                       (f"{v:.0f}%", "white" if v > mid else "#222222")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=PLOT_CFG["fontsize_cpg"],
                    fontfamily=PLOT_CFG["fontfamily"],
                    color=col, fontweight="bold")

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(
        [f"{g}\n[{GENE_ANNOT[g]}]" if g in GENE_ANNOT else g for g in genes],
        fontsize=PLOT_CFG["fontsize_tick"],
        fontfamily=PLOT_CFG["fontfamily"])
    for tick, g in zip(ax.get_xticklabels(), genes):
        tick.set_color(ANNOT_COLORS.get(GENE_ANNOT.get(g, ""), "#222222"))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=PLOT_CFG["fontsize_tick"],
                       fontfamily=PLOT_CFG["fontfamily"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% methylation", fontsize=PLOT_CFG["fontsize_cpg"],
                   fontfamily=PLOT_CFG["fontfamily"])
    cbar.ax.tick_params(labelsize=PLOT_CFG["fontsize_cpg"])

    annots = sorted({GENE_ANNOT[g] for g in genes if g in GENE_ANNOT})
    if annots:
        ax.legend(handles=[mpatches.Patch(color=ANNOT_COLORS.get(a, "#888888"),
                                          label=a) for a in annots],
                  fontsize=PLOT_CFG["fontsize_cpg"],
                  bbox_to_anchor=(1.18, 1.0), loc="upper left",
                  framealpha=0.7, facecolor="white", edgecolor="#cccccc",
                  title="Marker set",
                  title_fontsize=PLOT_CFG["fontsize_cpg"])

    window_txt = f"TSS +{promoter_window} bp" if promoter_window else "gene body"
    ax.set_title(f"Methylation at marker genes  ·  {cohort}  ·  {window_txt}",
                 fontsize=PLOT_CFG["fontsize_title"], fontweight="bold",
                 fontfamily=PLOT_CFG["fontfamily"], pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out_pdf}")

    tsv = f"{out_dir}/marker_gene_methylation{suffix}.tsv"
    mat.to_csv(tsv, sep="\t")
    print(f"[SAVED] {tsv}")


# ─── GENE-BODY vs PROMOTER SIDE-BY-SIDE ───────────────────────────────────────

def plot_window_comparison(cohort, samples, genes, out_dir,
                           promoter_window=1000, overwrite=False):
    """
    Heatmap with samples on Y and, for each gene, TWO columns:
    full gene body and the TSS promoter window. Makes the methylation "step"
    between body and promoter explicit (the thing a single-window heatmap hides).
    """
    os.makedirs(out_dir, exist_ok=True)
    gsafe = "_".join(g.replace("/", "-") for g in genes)
    out_pdf = f"{out_dir}/window_compare_{gsafe}_tss{promoter_window}.pdf"
    if os.path.exists(out_pdf) and not overwrite:
        print(f"[SKIP] exists: {out_pdf}")
        return

    labels = [short_label(s, cohort) for s in samples]
    # column order: (gene, "body"), (gene, "promoter") for each gene
    col_specs, col_titles = [], []
    for g in genes:
        col_specs.append((g, None));            col_titles.append(f"{g}\ngene body")
        col_specs.append((g, promoter_window)); col_titles.append(f"{g}\nTSS +{promoter_window}")

    mat = np.full((len(labels), len(col_specs)), np.nan)
    for j, (g, win) in enumerate(col_specs):
        chrom, start, end = gene_window(g, win)
        print(f"[INFO] {g} {'promoter' if win else 'body'}  {chrom}:{start:,}-{end:,}")
        for i, sid in enumerate(samples):
            val, n = mean_methylation_region(
                bedmethyl_path(cohort, sid), chrom, start, end)
            mat[i, j] = val
            print(f"    {labels[i]}: " +
                  (f"{val:.1f}%  (n={n})" if not np.isnan(val) else "NA"))

    if np.all(np.isnan(mat)):
        print("[ERROR] no data"); return
    vmin = max(0,   np.floor(np.nanmin(mat) / 5) * 5)
    vmax = min(100, np.ceil(np.nanmax(mat) / 5) * 5)

    fig, ax = plt.subplots(
        figsize=(1.5 * len(col_specs) + 3, 0.55 * len(labels) + 3),
        facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    mid = (vmin + vmax) / 2
    for i in range(len(labels)):
        for j in range(len(col_specs)):
            v = mat[i, j]
            txt, col = ("NA", "#aaaaaa") if np.isnan(v) else \
                       (f"{v:.0f}%", "white" if v > mid else "#222222")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=PLOT_CFG["fontsize_cpg"],
                    fontfamily=PLOT_CFG["fontfamily"], color=col,
                    fontweight="bold")

    ax.set_xticks(range(len(col_specs)))
    ax.set_xticklabels(col_titles, fontsize=PLOT_CFG["fontsize_tick"],
                       fontfamily=PLOT_CFG["fontfamily"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=PLOT_CFG["fontsize_tick"],
                       fontfamily=PLOT_CFG["fontfamily"])

    # divider between each gene's pair of columns
    for k in range(2, len(col_specs), 2):
        ax.axvline(k - 0.5, color="#444444", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% methylation", fontsize=PLOT_CFG["fontsize_cpg"],
                   fontfamily=PLOT_CFG["fontfamily"])
    cbar.ax.tick_params(labelsize=PLOT_CFG["fontsize_cpg"])

    ax.set_title(f"Gene body vs promoter methylation  ·  {cohort}",
                 fontsize=PLOT_CFG["fontsize_title"], fontweight="bold",
                 fontfamily=PLOT_CFG["fontfamily"], pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[SAVED] {out_pdf}")

    df = pd.DataFrame(mat, index=labels, columns=[f"{g}_{'prom' if w else 'body'}"
                                                  for g, w in col_specs])
    tsv = f"{out_dir}/window_compare_{gsafe}_tss{promoter_window}.tsv"
    df.to_csv(tsv, sep="\t")
    print(f"[SAVED] {tsv}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Promoter methylation heatmaps (cohort-agnostic)")
    p.add_argument("--cohort", required=True, help="Cohort ID, e.g. ONTWGS10")
    p.add_argument("--samples", nargs="*", default=None,
                   help="Optional subset of sample IDs")
    p.add_argument("--pattern", default=None, help="Regex filter on sample IDs")
    p.add_argument("--pair", nargs=2, metavar=("SAMPLE_A", "SAMPLE_B"),
                   help="Binned promoter density for one pair")
    p.add_argument("--pairs", choices=["all"],
                   help="Binned promoter density for every pair in the cohort")
    p.add_argument("--marker-heatmap", action="store_true",
                   help="Genes x samples marker methylation heatmap")
    p.add_argument("--window-compare", action="store_true",
                   help="Gene-body vs promoter side-by-side heatmap")
    p.add_argument("--genes", nargs="*", default=DEFAULT_MARKER_GENES,
                   help=f"Genes for the heatmap (default: {' '.join(DEFAULT_MARKER_GENES)})")
    p.add_argument("--promoter-window", type=int, default=None,
                   help="Use N bp from the TSS instead of the full gene body")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not (args.pair or args.pairs or args.marker_heatmap or args.window_compare):
        p.error("Provide --pair, --pairs all, --marker-heatmap, or --window-compare")

    samples = list_samples(args.cohort, samples=args.samples,
                           pattern=args.pattern)
    if not samples:
        sys.exit("[ERROR] No samples with a bedMethyl found.")

    out_dir = f"{out_base(args.cohort)}/plots/promoter_heatmap"
    os.makedirs(out_dir, exist_ok=True)

    if args.marker_heatmap:
        missing = [g for g in args.genes if g not in GENE_COORDS]
        if missing:
            sys.exit(f"[ERROR] not in gene_coords.py: {', '.join(missing)}")
        plot_marker_heatmap(args.cohort, samples, args.genes, out_dir,
                            promoter_window=args.promoter_window,
                            overwrite=args.overwrite)

    if args.window_compare:
        missing = [g for g in args.genes if g not in GENE_COORDS]
        if missing:
            sys.exit(f"[ERROR] not in gene_coords.py: {', '.join(missing)}")
        plot_window_comparison(args.cohort, samples, args.genes, out_dir,
                               promoter_window=args.promoter_window or 1000,
                               overwrite=args.overwrite)

    pairs = []
    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.pairs == "all":
        pairs = list(itertools.combinations(samples, 2))

    summary = []
    for a, b in pairs:
        res = plot_pair_density(args.cohort, a, b, out_dir,
                                overwrite=args.overwrite)
        if res:
            summary.append(res)
    if summary:
        tsv = f"{out_dir}/promoter_correlation_summary.tsv"
        pd.DataFrame(summary).to_csv(tsv, sep="\t", index=False)
        print(f"[SAVED] {tsv}")

    print("\nDone.")