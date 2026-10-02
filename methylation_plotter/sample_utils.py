"""
sample_utils.py
===============
Cohort-agnostic sample discovery for the modkit methylation pipeline.

No patient / condition / R-vs-S mapping. A "sample" is any directory under
    <OUT_ROOT>/<COHORT>/modkit/<SAMPLE_ID>/
that has a finished bedMethyl at
    3_methylation/<SAMPLE_ID>_CpG_5mC.bed[.gz]

Import in plotting scripts as:
    from sample_utils import out_base, list_samples, bedmethyl_path, short_label
"""

import os
import re
import glob
import subprocess

OUT_ROOT = os.environ.get(
    "MODKIT_OUT_ROOT",
    "/path/to/dir",
)

# Directories under <cohort>/modkit/ that are not samples
NON_SAMPLE_DIRS = {"dmr", "plots", "logs", "cache", "qc"}


def out_base(cohort):
    return f"{OUT_ROOT}/{cohort}/modkit"


def bedmethyl_path(cohort, sample_id, prefer_gz=True):
    """Return path to the sample's bedMethyl (.bed.gz preferred), or None."""
    bed = f"{out_base(cohort)}/{sample_id}/3_methylation/{sample_id}_CpG_5mC.bed"
    gz  = bed + ".gz"
    if prefer_gz and os.path.exists(gz):
        return gz
    if os.path.exists(bed):
        return bed
    if os.path.exists(gz):
        return gz
    return None


def list_samples(cohort, samples=None, pattern=None, verbose=True):
    """
    All sample IDs in a cohort with a finished bedMethyl, sorted.
      samples : optional explicit subset (list of sample IDs)
      pattern : optional regex applied to the sample ID
    """
    base = out_base(cohort)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Cohort output dir not found: {base}")

    ids = sorted(os.path.basename(p.rstrip("/"))
                 for p in glob.glob(f"{base}/*/"))
    ids = [s for s in ids if s not in NON_SAMPLE_DIRS
           and not s.startswith("cache")
           and not s.startswith("dmr")     # dmr, dmr_v1, ... are output dirs
           and not s.startswith("plots")]

    if samples:
        missing = [s for s in samples if s not in ids]
        if missing:
            print(f"[WARN] not found in {base}: {', '.join(missing)}")
        ids = [s for s in ids if s in samples]
    if pattern:
        rx = re.compile(pattern)
        ids = [s for s in ids if rx.search(s)]

    ready = []
    for s in ids:
        if bedmethyl_path(cohort, s):
            ready.append(s)
        elif verbose:
            print(f"[SKIP] no bedMethyl: {s}")

    if verbose:
        print(f"[INFO] {cohort}: {len(ready)} sample(s) ready")
    return ready


def short_label(sample_id, cohort=None):
    """ONTWGS9-12-FOO-NP01 -> 12-FOO   (plot-friendly label)."""
    lbl = sample_id
    if cohort and lbl.startswith(f"{cohort}-"):
        lbl = lbl[len(cohort) + 1:]
    lbl = re.sub(r"-NP\d+$", "", lbl)
    return lbl


def has_tabix(bed_path):
    return bed_path.endswith(".gz") and os.path.exists(bed_path + ".tbi")


def tabix_region(bed_path, chrom, start, end):
    """
    Stream a region from a bgzipped+indexed bedMethyl.
    Returns list of raw lines, or None if the file is not indexed.
    """
    if not has_tabix(bed_path):
        return None
    chrom_str = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    cmd = ["tabix", bed_path, f"{chrom_str}:{start}-{end}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return res.stdout.splitlines()


def sample_palette(samples, cmap_name="tab20"):
    """Stable sample -> color mapping (no condition semantics)."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    n = max(len(samples), 1)
    return {s: cmap(i % cmap.N) for i, s in enumerate(samples)}