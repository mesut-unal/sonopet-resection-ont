"""
methylation_browser.py
=======================
Per-patient, per-gene methylation browser plot.

Layout (all rows share x axis = genomic position):
  Row 1: % Methylation — Sonopet
  Row 2: % Methylation — Resection
  Row 3: Coverage      — Sonopet
  Row 4: Coverage      — Resection
  Row 5: CpG Islands

Usage:
  # One patient, one gene
  python methylation_browser.py --patient 1 --gene NF2

  # One patient, all genes
  python methylation_browser.py --patient 1 --gene all

  # All patients, one gene
  python methylation_browser.py --patient all --gene TERT

  # All patients, all genes
  python methylation_browser.py --patient all --gene all

Requirements:
  conda activate methylation_plots
  (bedtools not needed — reads bedMethyl directly)

CpG islands are fetched from UCSC once and cached locally.
"""

import os
import sys
import glob
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import gene coords config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_coords import GENE_COORDS

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Directory holding the ONT modkit sample folders and cpgIslandExt_hg38.txt.gz
OUT_BASE  = "/path/to/dir"
OUT_DIR   = f"{OUT_BASE}/plots/browser"
CACHE_DIR = f"{OUT_BASE}/plots/cache_browser"
GENOME    = "hg38"

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Minimum coverage to plot a CpG site
MIN_COVERAGE = 1

# Context window drawn on each side of the gene body. The plotted span is
# gene_start - FLANK_BP .. gene_end + FLANK_BP, so 2 Mb gives a ~4 Mb view and
# the gene itself is a thin sliver — it is shaded (see GENE_BAND_COLOR) so it
# stays findable. Set to 0 for the gene body alone.
FLANK_BP = 0

# Patient -> (R_key, S_key)
PATIENT_MAP = {
    1: (2,  1),
    3: (4,  5),
    4: (6,  7),
    5: (8,  9),
    6: (10, 11),
    7: (12, 13),
    8: (14, 15),
}

# Colors
COL_S     = "#5b9bd5"   # Sonopet  -- blue
COL_R     = "#ed7d31"   # Resection -- orange
COL_CPG   = "#2ecc71"   # CpG islands -- green
BG        = "#ffffff"
FG        = "#222222"
GENE_BAND = "#f2c14e"   # gene body shading when FLANK_BP widens the view

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def find_bedmethyl(key):
    pattern = f"{OUT_BASE}/ONTWGS9-{key}-*"
    matches = glob.glob(pattern)
    if not matches:
        return None
    sample_id = os.path.basename(matches[0])
    bed = f"{OUT_BASE}/{sample_id}/3_methylation/{sample_id}_CpG_5mC.bed"
    return bed if os.path.exists(bed) else None


def load_region(bed_path, chrom, start, end):
    """
    Load CpG sites from a bedMethyl file within a genomic region.
    Returns DataFrame with columns: pos, freq (0-100), coverage
    bedMethyl columns (0-based):
      0=chrom, 1=start, 2=end, 3=mod_code, 4=score, 5=strand,
      6=tStart, 7=tEnd, 8=color, 9=coverage, 10=freq(%), 11=N_mod,
      12=N_canon, 13=N_other, 14=N_del, 15=N_fail, 16=N_diff, 17=N_nocall
    """
    chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    rows = []

    opener = gzip.open if bed_path.endswith(".gz") else open
    with opener(bed_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 11:
                continue
            if f[0] != chrom_str:
                continue
            pos = int(f[1])
            if pos < start or pos > end:
                continue
            try:
                coverage = int(f[9])
                freq     = float(f[10])
            except ValueError:
                continue
            if coverage < MIN_COVERAGE:
                continue
            rows.append({"pos": pos, "freq": freq, "coverage": coverage})

    return pd.DataFrame(rows)


# CpG islands loaded once at startup from local file
_CPG_ISLANDS_DF = None

def _load_cpg_islands_file():
    """Load the full CpG island table into memory once."""
    global _CPG_ISLANDS_DF
    if _CPG_ISLANDS_DF is not None:
        return
    cpg_file = f"{OUT_BASE}/cpgIslandExt_hg38.txt.gz"
    if not os.path.exists(cpg_file):
        print(f"[WARN] CpG island file not found: {cpg_file}")
        _CPG_ISLANDS_DF = pd.DataFrame(columns=["chrom","start","end"])
        return
    print("[INFO] Loading CpG islands from local file...")
    # UCSC cpgIslandExt columns:
    # bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum, perCpg, perGc, obsExp
    df = pd.read_csv(cpg_file, sep="\t", header=None,
                     usecols=[1, 2, 3],
                     names=["chrom", "start", "end"])
    _CPG_ISLANDS_DF = df
    print(f"[INFO] Loaded {len(df):,} CpG islands")


def fetch_cpg_islands(chrom, start, end):
    """Return CpG islands overlapping a region as list of (start, end) tuples."""
    _load_cpg_islands_file()
    chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    mask = (
        (_CPG_ISLANDS_DF["chrom"] == chrom_str) &
        (_CPG_ISLANDS_DF["end"]   >  start) &
        (_CPG_ISLANDS_DF["start"] <  end)
    )
    islands = _CPG_ISLANDS_DF[mask]
    return list(zip(islands["start"], islands["end"]))


# ─── PLOT ─────────────────────────────────────────────────────────────────────

def plot_gene_patient(patient, gene, overwrite=False):
    coords = GENE_COORDS[gene][GENOME]
    chrom  = coords["chr"]
    gene_start = coords["start"]
    gene_end   = coords["end"]

    # plotted window = gene body + FLANK_BP on each side
    start = max(1, gene_start - FLANK_BP)
    end   = gene_end + FLANK_BP

    r_key, s_key = PATIENT_MAP[patient]
    r_label = f"P{patient}R"
    s_label = f"P{patient}S"

    r_bed = find_bedmethyl(r_key)
    s_bed = find_bedmethyl(s_key)

    if r_bed is None:
        print(f"[WARN] bedMethyl not found for {r_label} (key {r_key}), skipping")
        return
    if s_bed is None:
        print(f"[WARN] bedMethyl not found for {s_label} (key {s_key}), skipping")
        return

    print(f"[INFO] Patient {patient} | {gene} | chr{chrom}:{start:,}-{end:,}"
          f"  (gene {gene_start:,}-{gene_end:,}, flank {FLANK_BP:,} bp)")

    r_df = load_region(r_bed, chrom, start, end)
    s_df = load_region(s_bed, chrom, start, end)
    cpg_islands = fetch_cpg_islands(chrom, start, end)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        5, 1,
        figsize=(12, 10),
        facecolor=BG,
        gridspec_kw={"height_ratios": [3, 3, 2, 2, 0.8], "hspace": 0.08}
    )

    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.set_xlim(start, end)
        # With a multi-Mb flank the gene body is a fraction of a percent of the
        # width, so mark it on every row. A span narrower than ~0.3% of the
        # view renders as a hairline, so draw guide lines instead of a band.
        if FLANK_BP:
            if (gene_end - gene_start) / max(end - start, 1) >= 0.003:
                ax.axvspan(gene_start, gene_end, color=GENE_BAND, alpha=0.35,
                           zorder=0, linewidth=0)
            else:
                for x in (gene_start, gene_end):
                    ax.axvline(x, color=GENE_BAND, linewidth=1.0, alpha=0.9,
                               zorder=0)

    # ── Row 1: % Methylation Sonopet ──────────────────────────────────────────
    ax = axes[0]
    if not s_df.empty:
        ax.scatter(s_df["pos"], s_df["freq"], s=1.5, color=COL_S,
                   alpha=0.6, rasterized=True)
        # smoothed line
        s_sorted = s_df.sort_values("pos")
        ax.plot(s_sorted["pos"], s_sorted["freq"].rolling(20, center=True,
                min_periods=1).mean(), color=COL_S, linewidth=1.2, alpha=0.9)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("% Methylation", color=FG, fontsize=8, fontfamily="monospace")
    ax.set_title(
        f"Patient {patient}  ·  {gene}  ·  chr{chrom}:{start:,}–{end:,}"
        + (f"   (gene ±{FLANK_BP/1e6:g} Mb)" if FLANK_BP else ""),
        color="#e8e8e8", fontsize=11, fontweight="bold",
        fontfamily="monospace", pad=8
    )
    ax.text(0.01, 0.88, s_label, transform=ax.transAxes,
            color=COL_S, fontsize=9, fontfamily="monospace", fontweight="bold")
    ax.axhline(50, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.set_xticklabels([])

    # ── Row 2: % Methylation Resection ────────────────────────────────────────
    ax = axes[1]
    if not r_df.empty:
        ax.scatter(r_df["pos"], r_df["freq"], s=1.5, color=COL_R,
                   alpha=0.6, rasterized=True)
        r_sorted = r_df.sort_values("pos")
        ax.plot(r_sorted["pos"], r_sorted["freq"].rolling(20, center=True,
                min_periods=1).mean(), color=COL_R, linewidth=1.2, alpha=0.9)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("% Methylation", color=FG, fontsize=8, fontfamily="monospace")
    ax.text(0.01, 0.88, r_label, transform=ax.transAxes,
            color=COL_R, fontsize=9, fontfamily="monospace", fontweight="bold")
    ax.axhline(50, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.set_xticklabels([])

    # ── Row 3: Coverage Sonopet ───────────────────────────────────────────────
    ax = axes[2]
    if not s_df.empty:
        s_sorted = s_df.sort_values("pos")
        ax.fill_between(s_sorted["pos"], s_sorted["coverage"],
                        color=COL_S, alpha=0.5, linewidth=0)
        ax.plot(s_sorted["pos"], s_sorted["coverage"],
                color=COL_S, linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Coverage", color=FG, fontsize=8, fontfamily="monospace")
    ax.text(0.01, 0.82, s_label, transform=ax.transAxes,
            color=COL_S, fontsize=9, fontfamily="monospace", fontweight="bold")
    ax.set_xticklabels([])

    # ── Row 4: Coverage Resection ─────────────────────────────────────────────
    ax = axes[3]
    if not r_df.empty:
        r_sorted = r_df.sort_values("pos")
        ax.fill_between(r_sorted["pos"], r_sorted["coverage"],
                        color=COL_R, alpha=0.5, linewidth=0)
        ax.plot(r_sorted["pos"], r_sorted["coverage"],
                color=COL_R, linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Coverage", color=FG, fontsize=8, fontfamily="monospace")
    ax.text(0.01, 0.82, r_label, transform=ax.transAxes,
            color=COL_R, fontsize=9, fontfamily="monospace", fontweight="bold")
    ax.set_xticklabels([])

    # ── Row 5: CpG Islands ────────────────────────────────────────────────────
    ax = axes[4]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("CpG\nIslands", color=FG, fontsize=7,
                  fontfamily="monospace", labelpad=4)
    for istart, iend in cpg_islands:
        ax.add_patch(mpatches.Rectangle(
            (istart, 0.1), iend - istart, 0.8,
            facecolor=COL_CPG, edgecolor="none", alpha=0.85
        ))
    if not cpg_islands:
        ax.text(0.5, 0.5, "no CpG islands in region",
                transform=ax.transAxes, ha="center", va="center",
                color="#555555", fontsize=7, fontfamily="monospace")

    # x axis ticks on bottom panel only
    ax.tick_params(axis="x", colors=FG, labelsize=8)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_xlabel(f"chr{chrom} position (hg38)", color=FG,
                  fontsize=9, fontfamily="monospace")

    # ── save ──────────────────────────────────────────────────────────────────
    gene_safe = gene.replace("/", "_")
    # flank in the name so a narrow and a wide render of the same gene coexist
    flank_tag = f"_flank{FLANK_BP // 1000}kb" if FLANK_BP else ""
    out_pdf = f"{OUT_DIR}/patient{patient}_{gene_safe}{flank_tag}_browser.pdf"

    if os.path.exists(out_pdf) and not overwrite:
        print(f"  [SKIP] Already exists: {out_pdf}")
        return
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {out_pdf}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Methylation browser plot per patient per gene"
    )
    parser.add_argument(
        "--patient", required=True,
        help="Patient number or 'all'"
    )
    parser.add_argument(
        "--gene", required=True,
        help=f"Gene name or 'all'. Available: {', '.join(GENE_COORDS)}"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing plots (default: skip if already exists)"
    )
    args = parser.parse_args()

    patients = list(PATIENT_MAP.keys()) if args.patient == "all" \
               else [int(args.patient)]
    genes    = list(GENE_COORDS.keys()) if args.gene == "all" \
               else [args.gene]

    for g in genes:
        if g not in GENE_COORDS:
            print(f"[ERROR] Gene '{g}' not in gene_coords.py. "
                  f"Available: {', '.join(GENE_COORDS)}")
            sys.exit(1)

    for p in patients:
        if p not in PATIENT_MAP:
            print(f"[ERROR] Patient {p} not in PATIENT_MAP.")
            sys.exit(1)
        for g in genes:
            plot_gene_patient(p, g, overwrite=args.overwrite)

    print("\nDone.")
