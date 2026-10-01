"""
build_ont_matrix.py
===================
Rebuild ont_matrix.pkl from bedMethyl files — extracted from the
extract_ont_matrix() step of toronto_direct_mapping.py.

Only does the ONT extraction. Does NOT rebuild top_probes.pkl
(that comes from the Toronto EPIC reference and is unchanged).

Usage:
  conda activate methylation_plots
  python build_ont_matrix.py
"""

import os
import glob
import gzip
import numpy as np
import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(REPO_DIR, "input_data")

# Directory holding the ONT modkit sample folders (ONTWGS9-<key>-*/3_methylation/*_CpG_5mC.bed)
OUT_BASE  = "/path/to/modkit_output"

# Source of top_probes.pkl (unchanged — copy from the 6-patient cache)
SRC_CACHE = os.path.join(REPO_DIR, "cache_toronto_direct")

# Destination for the new 7-patient matrix
DST_CACHE = os.path.join(REPO_DIR, "cache_toronto_direct_7patients")

MANIFEST_FILE = os.path.join(INPUT_DIR, "EPIC_hg38_manifest.tsv.gz")

MIN_COVERAGE = 5     # min coverage in bedMethyl to use a CpG site

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

os.makedirs(DST_CACHE, exist_ok=True)


# ─── HELPERS (verbatim from toronto_direct_mapping.py) ───────────────────────

def load_manifest():
    """Load probe -> (chrom, pos) mapping from EPIC hg38 manifest."""
    print("[INFO] Loading EPIC manifest...")
    df = pd.read_csv(MANIFEST_FILE, sep="\t",
                     usecols=["probeID", "CpG_chrm", "CpG_beg"],
                     low_memory=False)
    df = df.dropna(subset=["CpG_chrm", "CpG_beg"])
    df["CpG_beg"] = df["CpG_beg"].astype(int)
    df = df[df["CpG_chrm"].str.match(r"^chr([0-9]+|X|Y)$")]
    df = df.set_index("probeID")
    print(f"[INFO] {len(df):,} probes with hg38 coordinates")
    return df


def find_bedmethyl(key):
    """Locate the bedMethyl file for a given ONTWGS9 sample key."""
    pattern = f"{OUT_BASE}/ONTWGS9-{key}-*"
    matches = glob.glob(pattern)
    if not matches:
        return None
    sample_id = os.path.basename(matches[0])
    bed = f"{OUT_BASE}/{sample_id}/3_methylation/{sample_id}_CpG_5mC.bed"
    return bed if os.path.exists(bed) else None


def load_ont_sample(bed_path, probe_positions):
    """Load ONT methylation at specific probe positions.
    probe_positions: dict chrom -> {pos -> probe_id}
    Returns Series: probe_id -> methylation fraction (0-1)"""
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1. top_probes.pkl — unchanged, just load it (and copy to new cache)
    src_top = f"{SRC_CACHE}/top_probes.pkl"
    dst_top = f"{DST_CACHE}/top_probes.pkl"
    if not os.path.exists(src_top):
        raise FileNotFoundError(f"top_probes.pkl not found: {src_top}")
    top_probes_df = pd.read_pickle(src_top)
    print(f"[INFO] top_probes: {top_probes_df.shape[0]:,} probes "
          f"x {top_probes_df.shape[1]} Toronto samples")
    if not os.path.exists(dst_top):
        top_probes_df.to_pickle(dst_top)
        print(f"[SAVED] {dst_top}")

    # 2. Manifest -> probe position lookup for the top probes only
    manifest_df = load_manifest()
    top_manifest = manifest_df.loc[
        manifest_df.index.intersection(top_probes_df.index)]
    probe_positions = {}
    for probe_id, row in top_manifest.iterrows():
        chrom = row["CpG_chrm"]
        pos   = row["CpG_beg"]
        probe_positions.setdefault(chrom, {})[pos] = probe_id
    n_pos = sum(len(v) for v in probe_positions.values())
    print(f"[INFO] Lookup covers {n_pos:,} probe positions")

    # 3. Scan each sample's bedMethyl
    series_list = []
    found, missing = [], []
    for key, meta in sorted(SAMPLE_META.items()):
        bed = find_bedmethyl(key)
        if bed is None:
            print(f"[WARN] bedMethyl NOT FOUND for key {key} ({meta['label']}) — skipping")
            missing.append(meta["label"])
            continue
        print(f"  [ONT] Extracting {meta['label']} (key {key})")
        s = load_ont_sample(bed, probe_positions).rename(meta["label"])
        print(f"        {len(s):,} probes with coverage >= {MIN_COVERAGE}")
        series_list.append(s)
        found.append(meta["label"])

    if not series_list:
        raise RuntimeError("No bedMethyl files found — check OUT_BASE and PATIENT_MAP")

    mat = pd.concat(series_list, axis=1)

    # 4. Save
    out = f"{DST_CACHE}/ont_matrix.pkl"
    mat.to_pickle(out)
    print(f"\n[INFO] ONT matrix: {mat.shape[0]:,} probes x {mat.shape[1]} samples")
    print(f"[INFO] Samples found  ({len(found)}): {', '.join(found)}")
    if missing:
        print(f"[WARN] Samples missing ({len(missing)}): {', '.join(missing)}")
    print(f"[INFO] NaN fraction: {mat.isna().sum().sum() / mat.size:.4f}")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()