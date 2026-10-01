"""
variations_analyzer.py — OOP library for somatic variation analysis.

Usage (Jupyter notebook)
------------------------
    from variations_analyzer_v2 import VariationAnalyzer, SVmethods, SNVmethods, CNVmethods

Classes
-------
VariationAnalyzer
    Shared utilities (PASS filter, sample discovery).

SVmethods
    All structural-variant (SV) logic:
    · summarize_sv_counts         — per-caller SV bar charts
    · extract_breakends           — build a breakend position index
    · overlap_curves_vs_reference — breakend overlap vs window size
    · compute_jaccard_matrix      — pairwise Jaccard similarity
    · compute_chrom_caller_overlap — per-chromosome overlap heatmap
    · build_breakend_index_per_sample / build_be_indices_union_and_percaller
    · compare_to_reference / print_similarity_table / plot_similarity_bars
    · plot_sample_lines / plot_overlap_curves / plot_overlap_dual_axis
    · plot_chrom_caller_heatmap / plot_jaccard_heatmap
    · nearest_distance_to_breakends

SNVmethods
    All small-variant / SNV logic:
    · get_smrest_vcfs / get_clair3_vcf / get_clairs_to_vcfs
    · load_small_variants / load_smrest_variants / load_mutect_maf / find_maf_for_sample
    · compute_longread_snv_overlap
    · filter_snvs / filter_snv_by_vaf
    · add_sv_distance_to_clairs_df / add_sv_distance_to_clairs_df_percaller
    · summarize_sv_proximity_for_callerset
    · compute_sv_proximity_per_caller_summary / compute_sv_proximity_per_caller_summary_smrest
    · plot_sv_proximity_dual_axis / plot_sv_proximity_dual_axis_to_only
    · build_proximity_locus_sets / collapse_snvs_by_proximity_window
    · call_mutation_islands / add_island_midpoint / annotate_island_support
    · plot_snvs_around_egfr / plot_long_events_around_egfr
    · plot_two_set_venn / plot_three_set_venn
    · parse_coral_bed / get_all_coral_amplicons / linearize_ecdna_segments
    · map_snv_to_linear / plot_ecdna_snvs / plot_all_ecdna_snvs_grid / debug_snv_positions
    · build_sv_all_for_egfr / plot_egfr_multi_track
    · build_chr_sample_count_matrix
    · count_Check

CNVmethods
    Placeholder for copy-number variation analysis.

CohortMethods
    Cohort-level (multi-sample / multi-patient) integration:
    · load_snv_cohort / load_sv_cohort — genome-wide, cached, parallel VCF loading
    · load_coral_amplicon_segments     — CORAL amplicon graphs as CN segments
    · load_methylation_cohort          — modkit bedMethyl -> region x sample betas
    · build_cohort_alterations         — gene x sample alteration matrix
    · plot_cohort_oncoprint            — layered SNV/INDEL/SV/CNV oncoprint
    · plot_cohort_burden / plot_cohort_chrom_heatmap
    · plot_cnv_frequency_genome / plot_sv_breakpoint_recurrence
    · plot_methylation_umap / plot_methylation_heatmap
    · plot_cohort_co_occurrence

Module-level constants
----------------------
CHROMS           : chromosomes 1-22, X, Y
PRIMARY_CHROMS   : same set as a set for fast lookup
SVTYPE_COLORS    : consistent color palette for SV categories
"""

import os, glob, tempfile, shutil
import pandas as pd
import vcfpy
import numpy as np
import pybedtools
import glob
import re
from collections import defaultdict
import tempfile, gzip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Iterable, Set
from itertools import combinations
import math
from pathlib import Path
import gzip
import pickle
import io
# multiprocessing import
from concurrent.futures import ProcessPoolExecutor
import joblib
import html as html_lib
from datetime import datetime

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]

SVTYPES: list[str] = ["INS", "BND", "DUP", "INV", "DEL"]
# Consistent colors per SV category (feel free to change hexes)
SVTYPE_COLORS = {
    "Total SV": "#8e8e8e",  # neutral gray
    "INS":      "#1f77b4",  # blue
    "BND":      "#ff7f0e",  # orange
    "DUP":      "#2ca02c",  # green
    "INV":      "#9467bd",  # purple
    "DEL":      "#d62728",  # red"
}

PRIMARY_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}

_BND_MATE_RE = re.compile(r'[\[\]](?P<chrom>[^:\[\]]+):(?P<pos>\d+)[\[\]]')

DEFAULT_CALLER_FOLDERS: dict[str, str] = {
    "nanomonsv":          "nanomonsv_out",
    "severus":            "severus",
    "sniffles_v1":        "sniffles1_dec10",
    "sniffles_v2":        "sniffles2_matchingDecoil",
    "manta_v1.6":         "manta",
    "manta":              "manta_from_peter",
    "savana":             "savana",
}

# ── gene coordinates (hg38) ──────────────────────────────────────────────────
# Chordoma genes live here; gene_coords.py is the separate meningioma table for
# the methylation browser. Both are GENCODE v44 gene spans, so the genes they
# share (CDKN2A/B, PIK3CA, PTEN, TP53, SMARCB1, EGFR, TERT) agree exactly —
# tests/check_gene_coords.py asserts that, since three hand-maintained tables
# previously gave three different answers for the same genes.
#
# Verified 2026-08-27 against GENCODE v44, the UCSC hg38 HGNC track and Ensembl
# GRCh38. TBXT and FN1 were badly wrong before: TBXT ran ~28.9 kb past the gene
# (72% of the window was not TBXT) and FN1 ran 157 kb past it (68% not FN1).
GENE_COORDS: dict[str, dict] = {
    "TBXT": {"chrom": "6",  "start": 166_157_656, "end": 166_168_700},
    "FN1":  {"chrom": "2",  "start": 215_360_440, "end": 215_436_073},
    "SOX9": {"chrom": "17", "start":  72_121_020, "end":  72_126_416},
}

class VariationAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def vcf_record_is_pass(rec) -> bool:
        """
        Return True if a vcfpy Record should be treated as PASS.

        Accept:
        - FILTER empty list / None
        - FILTER == ["PASS"]
        - FILTER == ["."]
        Reject:
        - any non-PASS / non-'.' filter flag
        """
        flt = rec.FILTER
        if flt in (None, [], ["PASS"], ["."]):
            return True
        # vcfpy uses a list of filter IDs
        return not any(f not in (".", "PASS") for f in flt)
    
    # Pattern that matches cordoma sample folder names (e.g. ONTWGS12-35-240548-NP01)
    SAMPLE_DIR_RE = re.compile(r'^ONTWGS\d+-\d+-\d+-NP\d+$')

    def discover_samples(root: str, pattern: "re.Pattern | None" = None) -> Set[str]:
        """
        Return immediate subfolder names under `root` that look like sample IDs.
        By default, only folders matching SAMPLE_DIR_RE are included, which filters
        out directories like mutational_signatures_output_vaf0.1_0.6, SBS, vcfs, etc.
        Pass pattern=None explicitly to disable filtering.
        """
        root = Path(root)
        if not root.exists():
            return set()

        pat = VariationAnalyzer.SAMPLE_DIR_RE if pattern is None else pattern
        return {
            p.name
            for p in root.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and pat.match(p.name)
        }

############################################
#### METHODS RELATED TO SV CALLERS ONLY ####
############################################
class SVmethods:
    @staticmethod
    def normalize_chrom(chrom: str) -> str:
        c = chrom.strip()
        if c.lower().startswith("chr"):
            c = c[3:]
        if c.upper() in {"M", "MT"}:
            return "MT"
        return c

    @staticmethod
    def is_primary_chrom(chrom: str) -> bool:
        return SVmethods.normalize_chrom(chrom) in PRIMARY_CHROMS

    @staticmethod
    def build_chrom_selection_set(chrom_selection) -> set[str] | None:
        """Normalise the user-supplied chrom_selection into a set of bare names."""
        if chrom_selection is None:
            return None
        if isinstance(chrom_selection, str):
            chrom_iter = [chrom_selection]
        else:
            chrom_iter = list(chrom_selection)
        return {SVmethods.normalize_chrom(c) for c in chrom_iter}

    @staticmethod
    def make_chrom_filter(
        chrom_selection_norm: set[str] | None,
        skip_non_primary: bool,
    ):
        """Return a callable  passes_chrom_filter(record_chrom) -> bool."""
        def passes(record_chrom: str) -> bool:
            n = SVmethods.normalize_chrom(record_chrom)
            if skip_non_primary and n not in PRIMARY_CHROMS:
                return False
            if chrom_selection_norm is not None and n not in chrom_selection_norm:
                return False
            return True
        return passes

    @staticmethod
    def get_sv_length(parts: list[str]) -> int | None:
        """
        Extract SV length from a split VCF record.
        1. SVLEN=<val> (absolute value).
        2. END=<val> − POS.
        Returns None if neither found (caller should let record pass).
        """
        if len(parts) < 8:
            return None
        info, pos_str = parts[7], parts[1]

        m = re.search(r'(?:^|;)SVLEN=([^;]+)', info)
        if m:
            try:
                return abs(int(m.group(1).split(",")[0]))
            except ValueError:
                pass

        m = re.search(r'(?:^|;)END=([^;]+)', info)
        if m:
            try:
                return abs(int(m.group(1)) - int(pos_str))
            except ValueError:
                pass

        return None

    @staticmethod
    def passes_length_filter(parts: list[str], min_sv_len: int) -> bool:
        """
        True when the record should be kept.
        BNDs (by ALT brackets or SVTYPE=BND) always pass.
        Records with unknown length pass conservatively.
        """
        if len(parts) > 4:
            alt = parts[4].upper()
            if "BND" in alt or "[" in alt or "]" in alt:
                return True

        if len(parts) > 7:
            m = re.search(r'(?:^|;)SVTYPE=([^;]+)', parts[7])
            if m and m.group(1).strip().upper() == "BND":
                return True

        sv_len = SVmethods.get_sv_length(parts)
        if sv_len is None:
            return True                       # unknown → conservative pass
        return sv_len >= min_sv_len

    @staticmethod
    def prefilter_vcf(
        path_in: str,
        passes_chrom_filter,                  # callable(chrom_str) -> bool
        min_sv_len: int = 10_000,
    ) -> str:
        """
        Write a temp VCF containing only records that:
        - pass the chromosome filter, AND
        - have |SVLEN| >= min_sv_len  (BNDs always pass; unknown length passes).

        Also performs minimal structural repairs:
        - lines missing FILTER column have 'PASS' inserted at position 6.
        - lines with wrong column count after repair are dropped.

        Returns the path to the temp file (caller must delete it).
        """
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="svcounts_", suffix=".vcf")
        os.close(tmp_fd)

        expected_fields: int | None = None

        with open(path_in, "rt", encoding="utf-8", errors="ignore") as fin, \
            open(tmp_path, "wt", encoding="utf-8") as fout:

            for raw in fin:
                line = raw.rstrip("\n")

                # --- header ---
                if line.startswith("#"):
                    fout.write(raw)
                    if line.startswith("#CHROM"):
                        hdr_cols = line.split("\t")
                        num_samples = max(0, len(hdr_cols) - 9)
                        expected_fields = 9 + num_samples
                    continue

                if "\t" not in line:
                    continue

                parts = line.split("\t")

                # chromosome filter
                if not passes_chrom_filter(parts[0].strip()):
                    continue

                # structural repair: insert missing FILTER column
                exp = expected_fields if expected_fields is not None else 10
                if len(parts) == exp - 1:
                    parts.insert(6, "PASS")

                if len(parts) != exp:
                    continue

                # length filter (applied after repair so INFO is at correct index)
                if not SVmethods.passes_length_filter(parts, min_sv_len):
                    continue

                fout.write("\t".join(parts) + "\n")

        return tmp_path

    @staticmethod
    def init_counts() -> dict[str, int]:
        return {"INS": 0, "BND": 0, "DUP": 0, "INV": 0, "DEL": 0, "total": 0}

    @staticmethod
    def count_svs_in_vcf(
        vcf_file: str,
        passes_chrom_filter,
        min_sv_len: int = 10_000,
    ) -> dict[str, int] | None:
        """
        Open *vcf_file*, pre-filter it, and return a counts dict.
        Returns None on unrecoverable read error.
        """
        def _open_filtered() -> tuple[vcfpy.Reader, str]:
            tmp = SVmethods.prefilter_vcf(vcf_file, passes_chrom_filter, min_sv_len)
            return vcfpy.Reader.from_path(tmp), tmp

        tmp_path: str | None = None
        try:
            reader, tmp_path = _open_filtered()
        except Exception as e:
            print(f"[WARN] Failed to open {vcf_file} after prefilter: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

        counts = SVmethods.init_counts()
        try:
            for rec in reader:
                if not SVmethods.vcf_record_is_pass(rec):
                    continue
                svt = rec.INFO.get("SVTYPE")
                if isinstance(svt, list):
                    svt = svt[0]
                if svt is None:
                    continue
                svt = str(svt).strip().upper()
                if svt in SVTYPES:
                    counts[svt] += 1
                counts["total"] += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return counts

    @staticmethod
    def vcf_record_is_pass(rec) -> bool:
        """Return True when the record's FILTER is PASS (or empty/missing)."""
        filters = rec.FILTER
        if not filters:
            return True
        return all(f.upper() in {"PASS", "."} for f in filters)

    @staticmethod
    def list_sample_dirs(parent_dir: str) -> list[str]:
        return sorted(d for d in glob.glob(os.path.join(parent_dir, "*"))
                    if os.path.isdir(d))

    @staticmethod
    def collect_vcfs_for_sample(caller: str, sample_dir: str) -> list[str]:
        if caller == "severus":
            som_dir = os.path.join(sample_dir, "somatic_SVs")
            if not os.path.isdir(som_dir):
                print(f"[WARN] Expected somatic_SVs not found: {som_dir}")
                return []
            return sorted(glob.glob(os.path.join(som_dir, "*.vcf")))
        return sorted(glob.glob(os.path.join(sample_dir, "*.vcf")))

    @staticmethod
    def build_filter_label(
        chrom_selection_norm: set[str] | None,
        skip_non_primary: bool,
        min_sv_len: int,
    ) -> str:
        """Human-readable description of the active filters, shown in plot titles."""
        parts: list[str] = []

        if chrom_selection_norm:
            def fmt(c: str) -> str:
                return "chrM" if c == "MT" else f"chr{c}"
            def ordkey(x: str):
                if x.isdigit(): return (0, int(x))
                if x == "X":    return (1, 23)
                if x == "Y":    return (1, 24)
                if x == "MT":   return (1, 25)
                return (2, 99)
            items = [fmt(c) for c in sorted(chrom_selection_norm, key=ordkey)]
            parts.append(", ".join(items))
        elif skip_non_primary:
            parts.append("primary chroms only")
        else:
            parts.append("all contigs")

        parts.append(f"SVLEN ≥ {min_sv_len:,} bp")

        return " | ".join(parts)

    @staticmethod
    def plot_caller_sv_counts(
        caller_name: str,
        data: dict[str, dict[str, int]],
        filter_label: str,
        save_fig: bool = False,
    ) -> None:
        """
        One figure per caller; one subplot per sample.
        *filter_label* is embedded in the figure super-title so readers know
        exactly which length and chromosome filters were applied.
        """
        if not data:
            print(f"[INFO] No data for {caller_name}")
            return

        samples = sorted(data.keys())
        n = len(samples)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
        axes = axes[0]

        cats = ["Total SV", "INS", "BND", "DUP", "INV", "DEL"]
        legend_handles = [mpatches.Patch(color=SVTYPE_COLORS[c], label=c) for c in cats]

        for idx, sample in enumerate(samples):
            ax = axes[idx]
            counts = data[sample]
            values = [
                counts.get("total", 0),
                counts.get("INS",   0),
                counts.get("BND",   0),
                counts.get("DUP",   0),
                counts.get("INV",   0),
                counts.get("DEL",   0),
            ]
            x = np.arange(len(cats))
            ax.bar(x, values, color=[SVTYPE_COLORS[c] for c in cats],
                edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=30, ha="right")
            ax.set_yscale("log")
            ax.set_ylabel("SV count")
            ax.set_title(sample)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        # ← filter details visible in every figure
        fig.suptitle(
            f"SV counts  •  {caller_name}\n"
            f"[filters: {filter_label}]",
            y=1.03, fontsize=_fs(13), fontweight="bold",
        )
        fig.tight_layout()
        fig.legend(handles=legend_handles, loc="upper right", ncol=3, frameon=False)

        if save_fig:
            out = f"{caller_name}_sv_counts.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved {out}")
        plt.show()

    @staticmethod
    def summarize_sv_counts(
        root_folder: str,
        chrom_selection=None,
        skip_non_primary: bool = True,
        caller_folders: dict | None = None,
        min_sv_len: int = 10_000,
        save_fig: bool = False,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """
        Scan *root_folder* for per-caller / per-sample VCF files,
        count SVs passing PASS filter + chromosome filter + length filter,
        and plot one figure per caller.

        Parameters
        ----------
        root_folder       : top-level directory containing one sub-dir per caller
        chrom_selection   : None → use all (subject to skip_non_primary);
                            str / list / set  → restrict to those chromosomes
        skip_non_primary  : drop GL/KI/chrUn/random/etc. contigs
        caller_folders    : override the default caller-name → sub-dir mapping
        min_sv_len        : minimum |SVLEN| (bp) to include; BNDs always included
        save_fig          : save PNG alongside showing it
        """
        if caller_folders is None:
            caller_folders = DEFAULT_CALLER_FOLDERS

        chrom_selection_norm = SVmethods.build_chrom_selection_set(chrom_selection)
        passes_chrom = SVmethods.make_chrom_filter(chrom_selection_norm, skip_non_primary)
        flabel = SVmethods.build_filter_label(chrom_selection_norm, skip_non_primary, min_sv_len)

        # { caller -> { sample_name -> counts_dict } }
        sample_counts: dict[str, dict[str, dict[str, int]]] = {
            caller: {} for caller in caller_folders
        }

        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(root_folder, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue

            sample_dirs = SVmethods.list_sample_dirs(caller_path)
            if not sample_dirs:
                print(f"[INFO] No sample directories under {caller_path}")
                continue

            for sample_dir in sample_dirs:
                sample_name = os.path.basename(sample_dir.rstrip(os.sep))
                vcfs = SVmethods.collect_vcfs_for_sample(caller, sample_dir)
                if not vcfs:
                    print(f"[INFO] No VCFs for {caller} / {sample_name}")
                    continue

                for vf in vcfs:
                    counts = SVmethods.count_svs_in_vcf(vf, passes_chrom, min_sv_len)
                    if counts is None:
                        continue
                    if sample_name not in sample_counts[caller]:
                        sample_counts[caller][sample_name] = SVmethods.init_counts()
                    for key in counts:
                        sample_counts[caller][sample_name][key] += counts[key]

        for caller in caller_folders:
            SVmethods.plot_caller_sv_counts(
                caller,
                sample_counts.get(caller, {}),
                flabel,
                save_fig=save_fig,
            )

        print("\n=== Summary ===")
        for caller in caller_folders:
            print(f"\nCaller: {caller}")
            for sample, counts in sample_counts.get(caller, {}).items():
                print(f"  {sample}: {counts}")

        return sample_counts

    # ---------- Similarity helpers ----------
    @staticmethod
    def _choose_ref(sample_counts, preferred="manta_from_peter", fallbacks=("manta",)):
        if preferred in sample_counts and sample_counts[preferred]:
            return preferred
        for c in fallbacks:
            if c in sample_counts and sample_counts[c]:
                return c
        raise ValueError("No suitable reference caller found.")

    @staticmethod
    def _shared_samples(sample_counts, caller_a, caller_b, normalizer=None):
        # normalizer can harmonize names if needed (default: identity)
        if normalizer is None:
            normalizer = lambda s: s
        sa = {normalizer(s) for s in sample_counts.get(caller_a, {}).keys()}
        sb = {normalizer(s) for s in sample_counts.get(caller_b, {}).keys()}
        return sorted(sa & sb)

    @staticmethod
    def _vectorize(sample_counts, caller, samples, key, normalizer=None):
        if normalizer is None:
            normalizer = lambda s: s
        # map from normalized -> original for this caller
        m = {}
        for s in sample_counts[caller].keys():
            m[normalizer(s)] = s
        vals = []
        for sn in samples:
            orig = m.get(sn, None)
            c = sample_counts[caller].get(orig, {}) if orig else {}
            vals.append(float(c.get(key, 0)))
        return np.array(vals, dtype=float)

    @staticmethod
    def _cosine_similarity(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def get_global_common_samples(sample_counts,
                                ref_caller,
                                ignore_callers=(),
                                sample_normalizer=None):
        """
        Return a sorted list of sample names (normalized) that are present in
        the reference caller AND in every other caller considered (excluding ignores).
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s

        # callers we will compare (ref + others, excluding ignored and empties)
        callers_in_scope = []
        for c, d in sample_counts.items():
            if c in ignore_callers:
                continue
            if not d:
                continue
            callers_in_scope.append(c)
        if ref_caller not in callers_in_scope:
            raise ValueError(f"Reference caller '{ref_caller}' has no samples.")

        common = None
        for c in callers_in_scope:
            s = {sample_normalizer(x) for x in sample_counts[c].keys()}
            common = s if common is None else (common & s)

        return sorted(common) if common else []

    @staticmethod
    def compare_to_reference(sample_counts,
                            preferred_ref="manta_from_peter",
                            fallback_refs=("manta",),
                            ignore_callers=("manta_v1.6",),
                            keys=("total","INS","BND","DUP","INV","DEL"),
                            sample_normalizer=None,
                            force_samples=None):
        """
        Returns: (ref_name, sim_dict) where sim_dict = {caller: {key: cosine}}
        If force_samples is provided (list of normalized names), comparisons use exactly
        those samples for ALL callers (intersection you computed globally).
        """
        ref = SVmethods._choose_ref(sample_counts, preferred=preferred_ref, fallbacks=fallback_refs)
        out = {}
        all_callers = [c for c in sample_counts.keys()
                    if c != ref and c not in set(ignore_callers)]
        for caller in all_callers:
            # Use global intersection if provided; else fall back to pairwise overlap
            if force_samples is not None:
                shared = force_samples
            else:
                shared = SVmethods._shared_samples(sample_counts, caller, ref, normalizer=sample_normalizer)

            if not shared:
                continue

            out[caller] = {}
            for k in keys:
                va = SVmethods._vectorize(sample_counts, caller, shared, k, normalizer=sample_normalizer)
                vr = SVmethods._vectorize(sample_counts, ref,    shared, k, normalizer=sample_normalizer)
                out[caller][k] = SVmethods._cosine_similarity(va, vr)
        return ref, out

    @staticmethod
    def print_similarity_table(sim_dict, ref_name):
        keys = ["total","INS","BND","DUP","INV","DEL"]
        for k in keys:
            rows = [(caller, vals[k]) for caller, vals in sim_dict.items() if k in vals]
            if not rows:
                continue
            rows.sort(key=lambda x: (np.nan_to_num(x[1], nan=-1.0)), reverse=True)
            print(f"\n=== Cosine similarity vs {ref_name} — {k} ===")
            for caller, val in rows:
                vs = "nan" if (val is None or np.isnan(val)) else f"{val:0.4f}"
                print(f"{caller:18s}  {vs}")

    @staticmethod
    def plot_similarity_bars(sim_dict, key="total", title_prefix="Cosine similarity"):
        """
        Bar chart of cosine similarity for a single SV category (key).
        Bars are colored using SVTYPE_COLORS to match your other plots.
        """
        # Map 'total' -> 'Total SV' for color lookup
        color_key = "Total SV" if key == "total" else key

        callers, vals = [], []
        for caller, d in sim_dict.items():
            v = d.get(key, np.nan)
            if not np.isnan(v):
                callers.append(caller)
                vals.append(v)
        if not callers:
            print(f"[INFO] No data to plot for key='{key}'")
            return

        order = np.argsort(vals)[::-1]
        callers = [callers[i] for i in order]
        vals    = [vals[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(6, 1.2*len(callers)), 4.5))
        bar_color = SVTYPE_COLORS.get(color_key, "#8e8e8e")  # default gray if missing
        ax.bar(np.arange(len(callers)), vals, color=bar_color, edgecolor="black", linewidth=0.5)

        ax.set_xticks(np.arange(len(callers)))
        ax.set_xticklabels(callers, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Cosine similarity (0–1)")
        ax.set_title(f"{title_prefix} — {key}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Optional: small legend with the SV type color
        ax.legend([key if key != "total" else "Total SV"], frameon=False, loc="upper left")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _norm_to_orig_map(sample_dict, sample_normalizer=None):
        """Build {normalized_name -> original_name} for a caller's sample dict."""
        if sample_normalizer is None:
            sample_normalizer = lambda s: s
        m = {}
        for s in sample_dict.keys():
            m[sample_normalizer(s)] = s
        return m

    @staticmethod
    def plot_sample_lines(sample_counts,
                        callers,
                        sv_key="total",                 # one of: "total","INS","BND","DUP","INV","DEL"
                        samples_norm=None,              # ordered list of normalized sample names (e.g., common_samples)
                        caller_colors=None,             # {"nanomonsv":"#...", ...}
                        sample_normalizer=None,
                        title_prefix="SV counts by sample",
                        ylog=True):
        """
        Draws a line chart with x = samples, y = counts, one line per caller.
        Uses ONLY the provided 'samples_norm' order (your common sample set).
        """
        if samples_norm is None or len(samples_norm) == 0:
            print("[INFO] No samples to plot."); return

        # default palette fallback if color missing
        default_palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        if caller_colors is None:
            caller_colors = {c: default_palette[i % len(default_palette)] for i, c in enumerate(sorted(callers))}

        # gather series
        x = np.arange(len(samples_norm))
        series = []
        labels = []
        colors = []

        for caller in callers:
            if caller not in sample_counts or not sample_counts[caller]:
                continue
            m = SVmethods._norm_to_orig_map(sample_counts[caller], sample_normalizer=sample_normalizer)
            y = []
            for s_norm in samples_norm:
                s_orig = m.get(s_norm)
                cdict = sample_counts[caller].get(s_orig, {}) if s_orig else {}
                y.append(float(cdict.get(sv_key if sv_key != "total" else "total", 0)))
            series.append(y)
            labels.append(caller)
            colors.append(caller_colors.get(caller, '#444444'))

        if not series:
            print("[INFO] Nothing to plot for", sv_key); return

        # plot
        fig, ax = plt.subplots(figsize=(max(7, 1.2*len(samples_norm)), 4.5))
        for ys, lab, col in zip(series, labels, colors):
            ax.plot(x, ys, marker="o", linewidth=2, markersize=5, label=lab, color=col)

        ax.set_xticks(x)
        ax.set_xticklabels(samples_norm, rotation=30, ha="right")
        ax.set_ylabel(f"{sv_key} count" if sv_key != "total" else "Total SV count")
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(f"{title_prefix} — {sv_key}")
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _normalize_chrom(chrom: str) -> str:
        c = chrom.strip()
        if c.lower().startswith("chr"):
            c = c[3:]
        if c.upper() in {"M", "MT"}:
            return "MT"
        return c

    @staticmethod
    def _is_primary_chrom(chrom: str) -> bool:
        return SVmethods._normalize_chrom(chrom) in PRIMARY_CHROMS

    @staticmethod
    def extract_breakends(
    root_folder: str,
    caller_folders: dict,
    chrom_selection=None,
    skip_non_primary: bool = True,
    min_sv_len: int | None = None,       # <-- add this
    ):  
        """
        Returns:
        breakends: {caller: {sample: {svtype: {chrom: sorted_unique_positions(np.ndarray[int])}}}}
        Notes:
        - Non-BND with INFO/END -> include POS and END.
        - BND -> include POS and mate parsed from ALT as separate breakends.
        - Deduplicates per (caller, sample, svtype, chrom).
        """

        # ---- selection normalization
        if chrom_selection is None:
            CHROM_SELECTION_NORM = None
        else:
            if isinstance(chrom_selection, str):
                chrom_iter = [chrom_selection]
            else:
                chrom_iter = list(chrom_selection)
            CHROM_SELECTION_NORM = {SVmethods._normalize_chrom(c) for c in chrom_iter}

        def _passes_chrom_filter(chrom: str) -> bool:
            n = SVmethods._normalize_chrom(chrom)
            if skip_non_primary and not SVmethods._is_primary_chrom(chrom):
                return False
            if CHROM_SELECTION_NORM is not None and n not in CHROM_SELECTION_NORM:
                return False
            return True

        # ---- robust prefilter (header + filtered records only; repairs 1-field-short lines)
        def _prefilter_vcf(path_in: str) -> str:
            """
            Create a temp (plain-text) VCF with:
            - header lines copied,
            - non-header records kept only if CHROM passes filters,
            - if exactly one field short vs expected, insert 'PASS' at FILTER,
            - drop lines with no tab or wrong width after repair.
            Gzip-aware (by extension).
            """
            is_gz = path_in.endswith(".gz")
            opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="ignore")) if is_gz \
                    else (lambda p: open(p, "rt", encoding="utf-8", errors="ignore"))

            fd, tmp_path = tempfile.mkstemp(prefix="be_pref_", suffix=".vcf")
            os.close(fd)

            expected_fields = None  # 9 + #samples from #CHROM header
            with opener(path_in) as fin, open(tmp_path, "wt", encoding="utf-8") as fout:
                for raw in fin:
                    line = raw.rstrip("\n")

                    if line.startswith("#"):
                        fout.write(raw)
                        if line.startswith("#CHROM"):
                            cols = line.split("\t")
                            num_samples = max(0, len(cols) - 9)
                            expected_fields = 9 + num_samples
                        continue

                    if "\t" not in line:
                        continue

                    parts = line.split("\t")
                    chrom_field = parts[0].strip()
                    if not _passes_chrom_filter(chrom_field):
                        continue

                    exp = expected_fields if expected_fields is not None else 10

                    # Repair common "missing FILTER" (one short)
                    if len(parts) == exp - 1:
                        parts.insert(6, "PASS")

                    if len(parts) != exp:
                        continue

                    # length filter (optional)
                    if min_sv_len is not None:
                        if not SVmethods.passes_length_filter(parts, min_sv_len):
                            continue

                    fout.write("\t".join(parts) + "\n")

            return tmp_path

        # ---- traversal helpers (same structure as your counter)
        def list_sample_dirs(parent_dir):
            return sorted([d for d in glob.glob(os.path.join(parent_dir, "*")) if os.path.isdir(d)])

        def collect_vcfs_for_sample(caller, sample_dir):
            if caller == "severus":
                som_dir = os.path.join(sample_dir, "somatic_SVs")
                if not os.path.isdir(som_dir):
                    print(f"[WARN] Expected somatic_SVs not found: {som_dir}")
                    return []
                return sorted(glob.glob(os.path.join(som_dir, "*.vcf")))  # add *.vcf.gz if needed
            else:
                return sorted(glob.glob(os.path.join(sample_dir, "*.vcf")))

        # ---- storage
        breakends = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
        # {caller}{sample}{svtype}{chrom} -> set(positions)

        # ---- read & extract with mandatory prefilter when filtering is active
        need_filter = skip_non_primary or (chrom_selection is not None) or (min_sv_len is not None)

        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(root_folder, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue

            for sample_dir in list_sample_dirs(caller_path):
                sample_name = os.path.basename(sample_dir.rstrip(os.sep))
                for vf in collect_vcfs_for_sample(caller, sample_dir):
                    to_read = vf
                    temp_created = False
                    try:
                        if need_filter:
                            to_read = _prefilter_vcf(vf)
                            temp_created = True
                        reader = vcfpy.Reader.from_path(to_read)
                    except Exception as e:
                        # One retry with prefilter if we didn't already
                        if not need_filter:
                            try:
                                to_read = _prefilter_vcf(vf)
                                temp_created = True
                                reader = vcfpy.Reader.from_path(to_read)
                            except Exception as e2:
                                print(f"[WARN] Failed to read {vf} even after prefilter: {e2}")
                                if temp_created and os.path.exists(to_read):
                                    os.remove(to_read)
                                continue
                        else:
                            print(f"[WARN] Failed to read {vf}: {e}")
                            if temp_created and os.path.exists(to_read):
                                os.remove(to_read)
                            continue

                    try:
                        for rec in reader:
                            # --- PASS filter ---
                            if not VariationAnalyzer.vcf_record_is_pass(rec):
                                continue
                            svt = rec.INFO.get("SVTYPE")
                            if isinstance(svt, list):
                                svt = svt[0]
                            if svt is None:
                                continue
                            svt = str(svt).strip().upper()

                            # 1) local breakend at POS
                            nchrom = SVmethods._normalize_chrom(rec.CHROM)
                            pos = int(rec.POS)
                            breakends[caller][sample_name][svt][nchrom].add(pos)

                            # 2) mate / END
                            if svt == "BND":
                                # parse mate from ALT text (safe for vcfpy’s Breakend too)
                                alt_txt = None
                                try:
                                    alts = rec.ALT if isinstance(rec.ALT, list) else [rec.ALT]
                                    alt_txt = str(alts[0].value if hasattr(alts[0], "value") else alts[0])
                                except Exception:
                                    pass
                                if alt_txt:
                                    m = _BND_MATE_RE.search(alt_txt)
                                    if m:
                                        mch = SVmethods._normalize_chrom(m.group("chrom"))
                                        mpos = int(m.group("pos"))
                                        if _passes_chrom_filter(mch):
                                            breakends[caller][sample_name][svt][mch].add(mpos)
                            else:
                                end = rec.INFO.get("END")
                                if end is not None:
                                    try:
                                        end = int(end)
                                        breakends[caller][sample_name][svt][nchrom].add(end)
                                    except Exception:
                                        pass
                    finally:
                        if temp_created and os.path.exists(to_read):
                            os.remove(to_read)

        # convert sets -> sorted arrays
        be_sorted = {}
        for caller, d1 in breakends.items():
            be_sorted[caller] = {}
            for sample, d2 in d1.items():
                be_sorted[caller][sample] = {}
                for svt, d3 in d2.items():
                    be_sorted[caller][sample][svt] = {}
                    for chrom, s in d3.items():
                        arr = np.array(sorted(s), dtype=np.int64)
                        be_sorted[caller][sample][svt][chrom] = arr
        return be_sorted
    
    @staticmethod
    def extract_breakends_svlengthfilter(
        root_folder: str,
        caller_folders: dict,
        chrom_selection=None,
        skip_non_primary: bool = True,
        min_sv_len: int = 10_000,
    ):
        """
        Identical to extract_breakends but filters out SVs with |SVLEN| < min_sv_len.
        BNDs always pass (no single-locus length).
        Unknown length (no SVLEN or END in INFO) is kept conservatively.

        Returns:
        breakends: {caller: {sample: {svtype: {chrom: sorted_unique_positions(np.ndarray[int])}}}}
        """
        import re

        # ---- selection normalization (unchanged)
        if chrom_selection is None:
            CHROM_SELECTION_NORM = None
        else:
            if isinstance(chrom_selection, str):
                chrom_iter = [chrom_selection]
            else:
                chrom_iter = list(chrom_selection)
            CHROM_SELECTION_NORM = {SVmethods._normalize_chrom(c) for c in chrom_iter}

        def _passes_chrom_filter(chrom: str) -> bool:
            n = SVmethods._normalize_chrom(chrom)
            if skip_non_primary and not SVmethods._is_primary_chrom(chrom):
                return False
            if CHROM_SELECTION_NORM is not None and n not in CHROM_SELECTION_NORM:
                return False
            return True

        # ---- prefilter with length filter
        def _prefilter_vcf(path_in: str) -> str:

            def get_sv_length(parts: list) -> int | None:
                if len(parts) < 8:
                    return None
                info = parts[7]
                m = re.search(r'(?:^|;)SVLEN=([^;]+)', info)
                if m:
                    try:
                        return abs(int(m.group(1).split(",")[0]))
                    except ValueError:
                        pass
                m = re.search(r'(?:^|;)END=([^;]+)', info)
                if m:
                    try:
                        return abs(int(m.group(1)) - int(parts[1]))
                    except ValueError:
                        pass
                return None

            def passes_length_filter(parts: list) -> bool:
                # BNDs: no single-locus length → always keep
                alt = parts[4].upper() if len(parts) > 4 else ""
                if "[" in alt or "]" in alt:
                    return True
                if len(parts) > 7:
                    m = re.search(r'(?:^|;)SVTYPE=([^;]+)', parts[7])
                    if m and m.group(1).strip().upper() == "BND":
                        return True
                sv_len = get_sv_length(parts)
                if sv_len is None:
                    return True  # unknown → keep conservatively
                return sv_len >= min_sv_len

            is_gz = path_in.endswith(".gz")
            opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="ignore")) if is_gz \
                    else (lambda p: open(p, "rt", encoding="utf-8", errors="ignore"))

            fd, tmp_path = tempfile.mkstemp(prefix="be_pref_", suffix=".vcf")
            os.close(fd)

            expected_fields = None
            with opener(path_in) as fin, open(tmp_path, "wt", encoding="utf-8") as fout:
                for raw in fin:
                    line = raw.rstrip("\n")

                    if line.startswith("#"):
                        fout.write(raw)
                        if line.startswith("#CHROM"):
                            cols = line.split("\t")
                            num_samples = max(0, len(cols) - 9)
                            expected_fields = 9 + num_samples
                        continue

                    if "\t" not in line:
                        continue

                    parts = line.split("\t")
                    chrom_field = parts[0].strip()
                    if not _passes_chrom_filter(chrom_field):
                        continue

                    exp = expected_fields if expected_fields is not None else 10

                    if len(parts) == exp - 1:
                        parts.insert(6, "PASS")

                    if len(parts) != exp:
                        continue

                    if not passes_length_filter(parts):
                        continue

                    fout.write("\t".join(parts) + "\n")

            return tmp_path

        # ---- traversal helpers (unchanged)
        def list_sample_dirs(parent_dir):
            return sorted([d for d in glob.glob(os.path.join(parent_dir, "*")) if os.path.isdir(d)])

        def collect_vcfs_for_sample(caller, sample_dir):
            if caller == "severus":
                som_dir = os.path.join(sample_dir, "somatic_SVs")
                if not os.path.isdir(som_dir):
                    print(f"[WARN] Expected somatic_SVs not found: {som_dir}")
                    return []
                return sorted(glob.glob(os.path.join(som_dir, "*.vcf")))
            else:
                return sorted(glob.glob(os.path.join(sample_dir, "*.vcf")))

        # ---- storage (unchanged)
        breakends = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))

        need_filter = skip_non_primary or (chrom_selection is not None)

        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(root_folder, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue

            for sample_dir in list_sample_dirs(caller_path):
                sample_name = os.path.basename(sample_dir.rstrip(os.sep))
                for vf in collect_vcfs_for_sample(caller, sample_dir):
                    to_read = vf
                    temp_created = False
                    try:
                        # Always prefilter in this variant (length filter always active)
                        to_read = _prefilter_vcf(vf)
                        temp_created = True
                        reader = vcfpy.Reader.from_path(to_read)
                    except Exception as e:
                        print(f"[WARN] Failed to read {vf}: {e}")
                        if temp_created and os.path.exists(to_read):
                            os.remove(to_read)
                        continue

                    try:
                        for rec in reader:
                            if not VariationAnalyzer.vcf_record_is_pass(rec):
                                continue
                            svt = rec.INFO.get("SVTYPE")
                            if isinstance(svt, list):
                                svt = svt[0]
                            if svt is None:
                                continue
                            svt = str(svt).strip().upper()

                            nchrom = SVmethods._normalize_chrom(rec.CHROM)
                            pos = int(rec.POS)
                            breakends[caller][sample_name][svt][nchrom].add(pos)

                            if svt == "BND":
                                alt_txt = None
                                try:
                                    alts = rec.ALT if isinstance(rec.ALT, list) else [rec.ALT]
                                    alt_txt = str(alts[0].value if hasattr(alts[0], "value") else alts[0])
                                except Exception:
                                    pass
                                if alt_txt:
                                    m = _BND_MATE_RE.search(alt_txt)
                                    if m:
                                        mch = SVmethods._normalize_chrom(m.group("chrom"))
                                        mpos = int(m.group("pos"))
                                        if _passes_chrom_filter(mch):
                                            breakends[caller][sample_name][svt][mch].add(mpos)
                            else:
                                end = rec.INFO.get("END")
                                if end is not None:
                                    try:
                                        end = int(end)
                                        breakends[caller][sample_name][svt][nchrom].add(end)
                                    except Exception:
                                        pass
                    finally:
                        if temp_created and os.path.exists(to_read):
                            os.remove(to_read)

        # convert sets -> sorted arrays (unchanged)
        be_sorted = {}
        for caller, d1 in breakends.items():
            be_sorted[caller] = {}
            for sample, d2 in d1.items():
                be_sorted[caller][sample] = {}
                for svt, d3 in d2.items():
                    be_sorted[caller][sample][svt] = {}
                    for chrom, s in d3.items():
                        arr = np.array(sorted(s), dtype=np.int64)
                        be_sorted[caller][sample][svt][chrom] = arr
        return be_sorted

    @staticmethod
    def _count_posdict(d):
        return int(sum(arr.size for arr in d.values())) if d else 0


    @staticmethod
    def _hit_stats(ref_dict, test_dict, w_bp: int):
        ref_total = 0
        ref_hit = 0

        for chrom, ref_pos in ref_dict.items():
            if ref_pos.size == 0:
                continue
            ref_total += ref_pos.size

            test_pos = test_dict.get(chrom)
            if test_pos is None or test_pos.size == 0:
                continue

            for p in ref_pos:
                lo = np.searchsorted(test_pos, p - w_bp, side="left")
                hi = np.searchsorted(test_pos, p + w_bp, side="right")
                if hi > lo:
                    ref_hit += 1

        if ref_total == 0:
            return 0, 0, np.nan
        return int(ref_hit), int(ref_total), ref_hit / ref_total


    @staticmethod
    def overlap_curves_vs_reference(
        be_sorted,
        ref_caller="manta",
        ignore_callers=("manta_v1.6",),
        windows=(0,10,25,50,100,200,500,1000),
        svtypes=("BND","DEL","DUP","INV","INS"),
        allowed_samples=None,
        sample_normalizer=None,
    ):
        """
        Returns:
        stats: {caller: {key: {
                    "fraction": {w: float},
                    "hit":      {w: int},
                    "ref_total": int,
                    "test_total": int
                }}}
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s
        allowed_set = set(allowed_samples) if allowed_samples is not None else None
        svset = set(svtypes)
        ignore_set = set(ignore_callers) if ignore_callers else set()

        # Inline aggregation: {caller: {svt: {chrom: np.ndarray(sorted)}}}
        out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for caller, sd in be_sorted.items():
            for sample, svdict in sd.items():
                s_norm = sample_normalizer(sample)
                if allowed_set is not None and s_norm not in allowed_set:
                    continue
                for svt, chdict in svdict.items():
                    if svt not in svset:
                        continue
                    for chrom, arr in chdict.items():
                        if arr is not None and arr.size:
                            out[caller][svt][chrom].append(arr)

        agg = {}
        for caller, d1 in out.items():
            agg[caller] = {}
            for svt, d2 in d1.items():
                agg[caller][svt] = {}
                for chrom, lst in d2.items():
                    if not lst:
                        continue
                    cat = np.concatenate(lst)
                    if cat.size:
                        cat.sort()
                        agg[caller][svt][chrom] = cat

        if ref_caller not in agg:
            raise ValueError(f"Reference caller '{ref_caller}' has no breakends after filtering.")

        stats = {}
        for caller, cdict in agg.items():
            if caller == ref_caller or caller in ignore_set:
                continue

            stats[caller] = {}

            # per SVTYPE
            for svt in svtypes:
                ref = agg[ref_caller].get(svt, {})
                test = cdict.get(svt, {})

                stats[caller][svt] = {
                    "fraction": {},
                    "hit": {},
                    "ref_total": SVmethods._count_posdict(ref),
                    "test_total": SVmethods._count_posdict(test),
                }
                for w in windows:
                    hit, tot, frac = SVmethods._hit_stats(ref, test, w)
                    stats[caller][svt]["hit"][w] = hit
                    stats[caller][svt]["fraction"][w] = frac

            # total = union across svtypes
            ref_total = {}
            test_total = {}

            for svt in svtypes:
                for chrom, arr in agg[ref_caller].get(svt, {}).items():
                    ref_total.setdefault(chrom, []).append(arr)
                for chrom, arr in cdict.get(svt, {}).items():
                    test_total.setdefault(chrom, []).append(arr)

            ref_total = {ch: (np.concatenate(lst) if lst else np.array([], dtype=np.int64))
                        for ch, lst in ref_total.items()}
            test_total = {ch: (np.concatenate(lst) if lst else np.array([], dtype=np.int64))
                        for ch, lst in test_total.items()}

            for ch in ref_total:
                ref_total[ch].sort()
            for ch in test_total:
                test_total[ch].sort()

            stats[caller]["total"] = {
                "fraction": {},
                "hit": {},
                "ref_total": SVmethods._count_posdict(ref_total),
                "test_total": SVmethods._count_posdict(test_total),
            }
            for w in windows:
                hit, tot, frac = SVmethods._hit_stats(ref_total, test_total, w)
                stats[caller]["total"]["hit"][w] = hit
                stats[caller]["total"]["fraction"][w] = frac

        return stats

    @staticmethod
    def print_overlap_table(stats, w_focus=100, ref_name="manta"):
        keys = ["total","INS","BND","DUP","INV","DEL"]
        print(f"\n=== Overlap vs {ref_name} at ±{w_focus} bp (fraction; hit/ref_total; test_total) ===")

        for k in keys:
            rows = []
            for caller, d in stats.items():
                if k not in d:
                    continue
                frac = d[k]["fraction"].get(w_focus, np.nan)
                hit  = d[k]["hit"].get(w_focus, 0)
                rtot = d[k]["ref_total"]
                ttot = d[k]["test_total"]
                rows.append((caller, frac, hit, rtot, ttot))

            if not rows:
                continue

            rows.sort(key=lambda x: (np.nan_to_num(x[1], nan=-1.0)), reverse=True)
            print(f"\n-- {k} --")
            for caller, frac, hit, rtot, ttot in rows:
                fs = "nan" if (frac is None or np.isnan(frac)) else f"{frac:0.3f}"
                print(f"{caller:18s}  {fs}   ({hit}/{rtot})   test_total={ttot}")

    @staticmethod
    def plot_overlap_curves(stats, key="total", metric="fraction",
                            title_prefix="Breakend overlaps with manta",
                            ref_name="manta", caller_colors=None, ylog=False):
        """
        metric: "fraction" or "hit"
        """
        default_palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

        xs = None
        series = []
        labels = []

        for caller in sorted(stats.keys()):
            d = stats[caller]
            if key not in d:
                continue
            ws = sorted(d[key][metric].keys())
            ys = [d[key][metric][w] for w in ws]
            if xs is None:
                xs = ws
            series.append(ys)
            labels.append(caller)

        if not series:
            print(f"[INFO] No data to plot for key='{key}', metric='{metric}'")
            return

        if caller_colors is None:
            caller_colors = {c: default_palette[i % len(default_palette)]
                            for i, c in enumerate(labels)}

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (ys, lab) in enumerate(zip(series, labels)):
            col = caller_colors.get(lab, default_palette[i % len(default_palette)])
            ax.plot(xs, ys, marker="o", label=lab, color=col, linewidth=2, markersize=5)

        ax.set_xlabel("Window (±bp)")
        if metric == "fraction":
            ax.set_ylabel("Overlap fraction (ref breakends hit)")
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylabel("Hit count (#ref breakends hit)")

        ax.set_title(f"{title_prefix} — {key} — {metric}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False)

        if ylog:
            ax.set_yscale("log")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_total_records(stats, key="total", title_prefix="Total breakend records",
                        caller_colors=None):
        callers = [c for c in sorted(stats.keys()) if key in stats[c]]
        if not callers:
            print(f"[INFO] No data to plot for key='{key}'")
            return

        ref_totals  = [stats[c][key]["ref_total"]  for c in callers]
        test_totals = [stats[c][key]["test_total"] for c in callers]

        default_palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        if caller_colors is None:
            caller_colors = {c: default_palette[i % len(default_palette)]
                            for i, c in enumerate(callers)}

        x = np.arange(len(callers))
        width = 0.42

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - width/2, ref_totals,  width, label="ref_total")
        ax.bar(x + width/2, test_totals, width, label="test_total")

        ax.set_xticks(x)
        ax.set_xticklabels(callers, rotation=30, ha="right")
        ax.set_ylabel("Breakend count")
        ax.set_title(f"{title_prefix} — {key}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_overlap_dual_axis(stats, key="total",
                                        title_prefix="Breakend overlaps with manta within a window",
                                        ref_name="manta", caller_colors=None):
        """
        One line per caller:
        - line is hit count vs window (left y)
        - right y axis is fraction = hit / ref_total (no extra lines)
        """
        default_palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

        callers = [c for c in sorted(stats.keys()) if key in stats[c]]
        if not callers:
            print(f"[INFO] No data to plot for key='{key}'")
            return

        # windows = union across callers
        wset = set()
        for c in callers:
            wset.update(stats[c][key]["hit"].keys())
        xs = sorted(wset)
        if not xs:
            print(f"[INFO] No windows to plot for key='{key}'")
            return

        # ref_total should be identical across callers for this key; take from first
        ref_total = stats[callers[0]][key].get("ref_total", 0)
        if not ref_total:
            print(f"[INFO] ref_total is 0 for key='{key}' (cannot scale to fraction).")
            ref_total = None

        # colors
        if caller_colors is None:
            caller_colors = {c: default_palette[i % len(default_palette)]
                            for i, c in enumerate(callers)}

        fig, ax = plt.subplots(figsize=(7.8, 4.8))

        for i, caller in enumerate(callers):
            col = caller_colors.get(caller, default_palette[i % len(default_palette)])
            hits = [stats[caller][key]["hit"].get(w, np.nan) for w in xs]
            ax.plot(xs, hits, marker="o", linewidth=2, markersize=5, label=caller, color=col)

        ax.set_xlabel("Window (±bp)")
        ax.set_ylabel("# of overlapping records")
        ax.set_title(f"{title_prefix} — {key}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False)

        # Secondary axis showing fraction = hit / ref_total
        if ref_total is not None:
            secax = ax.secondary_yaxis(
                "right",
                functions=(lambda y: y / ref_total, lambda y: y * ref_total)
            )
            secax.set_ylabel("Overlap fraction (match / ref. total)")
            secax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _count_posdict(d):
        return int(sum(arr.size for arr in d.values())) if d else 0
    
    @staticmethod
    def _hit_stats(ref_dict, test_dict, w_bp: int):
        ref_total = 0
        ref_hit = 0

        for chrom, ref_pos in ref_dict.items():
            if ref_pos.size == 0:
                continue
            ref_total += ref_pos.size

            test_pos = test_dict.get(chrom)
            if test_pos is None or test_pos.size == 0:
                continue

            for p in ref_pos:
                lo = np.searchsorted(test_pos, p - w_bp, side="left")
                hi = np.searchsorted(test_pos, p + w_bp, side="right")
                if hi > lo:
                    ref_hit += 1

        if ref_total == 0:
            return 0, 0, np.nan
        return int(ref_hit), int(ref_total), ref_hit / ref_total

    @staticmethod
    def overlap_curves_R_vs_S(
        be_sorted,
        caller="sniffles_v1",
        windows=(0, 10, 25, 50, 100, 200, 500, 1000),
        svtypes=("BND", "DEL", "DUP", "INV", "INS"),
        save_fig=False,
    ):
        """
        For each SV type (+ total): one figure with one subplot per patient.
        Each subplot: solid line = R→S overlap, dashed = S→R overlap.
        """

        SEQID_TO_SR = {
            1:  "1S",  2:  "1R",
            3:  "2S",
            4:  "3R",  5:  "3S",
            6:  "4R",  7:  "4S",
            8:  "5R",  9:  "5S",
            10: "6R",  11: "6S",
            12: "7R",  13: "7S",
            14: "8R",  15: "8S",
        }

        def extract_seqid(sample_name):
            return int(sample_name.split("-")[1])

        patient_pairs = defaultdict(dict)
        for seqid, sr in SEQID_TO_SR.items():
            patient = int(sr[:-1])
            status  = sr[-1]
            patient_pairs[patient][status] = seqid
        patient_pairs = {p: v for p, v in patient_pairs.items() if "S" in v and "R" in v}

        caller_data = be_sorted.get(caller, {})
        if not caller_data:
            print(f"[WARN] No data for caller '{caller}'")
            return

        seqid_to_sample = {}
        for sname in caller_data:
            try:
                seqid_to_sample[extract_seqid(sname)] = sname
            except (IndexError, ValueError):
                pass

        valid_patients = []
        for patient, ids in sorted(patient_pairs.items()):
            sname_R = seqid_to_sample.get(ids["R"])
            sname_S = seqid_to_sample.get(ids["S"])
            if sname_R is None or sname_S is None:
                print(f"[INFO] Patient {patient}: missing R or S in '{caller}', skipping.")
                continue
            valid_patients.append((patient, sname_R, sname_S))

        if not valid_patients:
            print("[WARN] No complete R/S pairs found.")
            return

        def _agg_sample(sname, svt_list):
            tmp = defaultdict(list)
            svdict = caller_data.get(sname, {})
            for svt in svt_list:
                for chrom, arr in svdict.get(svt, {}).items():
                    if arr is not None and arr.size:
                        tmp[chrom].append(arr)
            result = {}
            for chrom, lst in tmp.items():
                cat = np.concatenate(lst)
                cat.sort()
                result[chrom] = cat
            return result

        xs      = list(windows)
        keys    = list(svtypes) + ["total"]
        ncols   = 4
        n       = len(valid_patients)
        nrows   = math.ceil(n / ncols)

        # Color per SV type, reuse your SVTYPE_COLORS
        key_colors = {**SVTYPE_COLORS, "total": SVTYPE_COLORS["Total SV"]}

        for key in keys:
            fig, axes = plt.subplots(nrows, ncols,
                                    figsize=(5 * ncols, 4 * nrows),
                                    squeeze=False)
            col = key_colors.get(key, "#8e8e8e")

            for idx, (patient, sname_R, sname_S) in enumerate(valid_patients):
                ax = axes[idx // ncols][idx % ncols]

                svt_list   = list(svtypes) if key == "total" else [key]
                posdict_R  = _agg_sample(sname_R, svt_list)
                posdict_S  = _agg_sample(sname_S, svt_list)

                if posdict_R and posdict_S:
                    hits_RvS = [SVmethods._hit_stats(posdict_R, posdict_S, w)[0] for w in windows]
                    hits_SvR = [SVmethods._hit_stats(posdict_S, posdict_R, w)[0] for w in windows]

                    ax.plot(xs, hits_RvS, marker="o", linewidth=2, markersize=5,
                            linestyle="-",  color=col, label="R → S")
                    ax.plot(xs, hits_SvR, marker="s", linewidth=2, markersize=5,
                            linestyle="--", color=col, label="S → R", alpha=0.7)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")

                ax.set_title(f"Patient {patient}", fontweight="bold")
                ax.set_xlabel("Window (±bp)")
                ax.set_ylabel("# overlapping breakends")
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(frameon=False, fontsize=_fs(8))

            # Hide unused panels
            for idx in range(n, nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            fig.suptitle(f"R vs S breakend overlap — {key}  |  {caller}",
                        fontsize=_fs(13), fontweight="bold")
            fig.tight_layout()

            if save_fig:
                out = f"{caller}_overlap_RvsS_{key}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                print(f"[OK] Saved {out}")
            plt.show()

    # --- helper: greedy intersection size for two sorted integer arrays with ±w matching
    @staticmethod
    def _intersection_size_with_window(a: np.ndarray, b: np.ndarray, w: int) -> int:
        """
        Count unique matches between two sorted integer arrays where |a[i] - b[j]| <= w.
        Greedy two-pointer; each element can be matched at most once.
        """
        if a.size == 0 or b.size == 0:
            return 0
        i = j = hits = 0
        while i < a.size and j < b.size:
            diff = a[i] - b[j]
            if diff < -w:
                i += 1
            elif diff > w:
                j += 1
            else:
                # match
                hits += 1
                i += 1
                j += 1
        return hits

    @staticmethod
    def _concat_total_per_chrom(agg_for_caller, svtypes):
        """
        Build {chrom: sorted np.ndarray} for TOTAL by concatenating all svtypes.
        """
        out = {}
        for svt in svtypes:
            for chrom, arr in agg_for_caller.get(svt, {}).items():
                out.setdefault(chrom, [])
                out[chrom].append(arr)
        for chrom, parts in out.items():
            if parts:
                cat = np.concatenate(parts)
                cat.sort()
                out[chrom] = cat
            else:
                out[chrom] = np.array([], dtype=np.int64)
        return out

    @staticmethod
    def compute_jaccard_matrix(
        be_sorted,                         # from extract_breakends(...)
        svtype="BND",                      # one of {"BND","DEL","DUP","INV","INS","total"}
        window_bp=100,
        callers=None,                      # list of callers to include; default = all in be_sorted
        ignore_callers=("manta_v1.6",),    # skip these
        allowed_samples=None,              # normalized common samples (e.g., common_samples)
        sample_normalizer=None,            # e.g., drop_trailing_cell
        svtypes_all=("BND","DEL","DUP","INV","INS"),
    ):
        """
        Returns (callers_kept, jaccard_matrix) for the given svtype and window.
        J(A,B) = |A ∩ B| / |A ∪ B| with ∩ computed by ±window matching, per-chrom summed.
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s

        # pick callers
        all_callers = sorted(be_sorted.keys()) if callers is None else list(callers)
        callers_kept = [c for c in all_callers if c not in set(ignore_callers)]

        # aggregate breakends across allowed samples using your existing aggregator
        # (requires the version that accepts allowed_samples + sample_normalizer)
        agg = SVmethods._aggregate_breakends(
            be_sorted,
            svtypes=set(svtypes_all),
            allowed_samples=allowed_samples,
            sample_normalizer=sample_normalizer
        )

        # prepare per-caller dict: {chrom: sorted np.ndarray}
        per = {}
        for c in callers_kept:
            if c not in agg:
                continue
            if svtype == "total":
                per[c] = SVmethods._concat_total_per_chrom(agg[c], svtypes_all)
            else:
                per[c] = {ch: arr for ch, arr in agg[c].get(svtype, {}).items()}

        # callers that actually have data
        callers_kept = [c for c in callers_kept if c in per]

        n = len(callers_kept)
        J = np.full((n, n), np.nan, dtype=float)

        # compute pairwise Jaccard
        for i in range(n):
            Ai = per[callers_kept[i]]
            # precompute sizes
            size_i = sum(arr.size for arr in Ai.values())
            for j in range(i, n):
                Aj = per[callers_kept[j]]
                size_j = sum(arr.size for arr in Aj.values())
                if size_i == 0 and size_j == 0:
                    J[i, j] = J[j, i] = np.nan
                    continue
                # intersection across chromosomes (only where both have data)
                inter = 0
                shared_chroms = set(Ai.keys()) & set(Aj.keys())
                for ch in shared_chroms:
                    a = Ai[ch]
                    b = Aj[ch]
                    if a.size and b.size:
                        inter += SVmethods._intersection_size_with_window(a, b, window_bp)
                union = size_i + size_j - inter
                J_val = (inter / union) if union > 0 else np.nan
                J[i, j] = J[j, i] = J_val

        return callers_kept, J

    @staticmethod
    def plot_jaccard_heatmap(callers, J, title="", cmap="Reds", vmin=0.0, vmax=1.0, annotate=True):
        """
        Simple heatmap with values in [0,1]. NaNs are masked.
        """
        import numpy.ma as ma
        M = ma.masked_invalid(J)

        fig, ax = plt.subplots(figsize=(1.1*len(callers), 1.0*len(callers)))
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(callers)))
        ax.set_yticks(np.arange(len(callers)))
        ax.set_xticklabels(callers, rotation=45, ha="right")
        ax.set_yticklabels(callers)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("")

        cmap = plt.cm.Reds.copy()
        cmap.set_under("white")
        plt.imshow(M, cmap=cmap, vmin=0.0001, vmax=1)


        if annotate:
            for i in range(len(callers)):
                for j in range(len(callers)):
                    if not np.isnan(J[i, j]):
                        ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", color="white" if J[i,j] > 0.5 else "black", fontsize=_fs(8))

        plt.tight_layout()
        plt.show()


    @staticmethod
    def _overlap_fraction_per_chrom(ref_dict, test_dict, w_bp: int):
        """
        For each chromosome, compute fraction of ref breakends that have ≥1 match
        in test within ±w. Returns {chrom: frac or np.nan if ref has 0}.
        """
        out = {}
        for chrom, ref_pos in ref_dict.items():
            ref_total = ref_pos.size
            if ref_total == 0:
                out[chrom] = np.nan
                continue
            test_pos = test_dict.get(chrom, np.array([], dtype=np.int64))
            if test_pos.size == 0:
                out[chrom] = 0.0
                continue
            hits = 0
            for p in ref_pos:
                lo = np.searchsorted(test_pos, p - w_bp, side="left")
                hi = np.searchsorted(test_pos, p + w_bp, side="right")
                if hi > lo:
                    hits += 1
            out[chrom] = hits / ref_total
        return out

    @staticmethod
    def _aggregate_breakends(bes, svtypes=None, allowed_samples=None, sample_normalizer=None):
        """
        Aggregate breakend arrays across *allowed* samples for each caller and SVTYPE.
        If allowed_samples is None -> include all samples.
        Returns: {caller: {svt: {chrom: np.ndarray positions}}}
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s
        allowed_set = set(allowed_samples) if allowed_samples is not None else None

        out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for caller, sd in bes.items():
            for sample, svdict in sd.items():
                s_norm = sample_normalizer(sample)
                if allowed_set is not None and s_norm not in allowed_set:
                    continue
                for svt, chdict in svdict.items():
                    if svtypes and svt not in svtypes:
                        continue
                    for chrom, arr in chdict.items():
                        if arr is not None and arr.size:
                            out[caller][svt][chrom].append(arr)

        # concat & sort
        agg = {}
        for caller, d1 in out.items():
            agg[caller] = {}
            for svt, d2 in d1.items():
                agg[caller][svt] = {}
                for chrom, lst in d2.items():
                    if not lst:
                        continue
                    cat = np.concatenate(lst)
                    if cat.size:
                        cat.sort()
                        agg[caller][svt][chrom] = cat
        return agg

    @staticmethod
    def _overlap_stats_per_chrom(ref_per, test_per, w_bp: int):
        """
        ref_per/test_per: {chrom: np.ndarray(sorted positions)}
        Returns dicts:
        hit_per[chrom]   = #ref breakends hit within window
        tot_per[chrom]   = #ref breakends total on chrom
        frac_per[chrom]  = hit / total  (NaN if total==0)
        """
        hit_per = {}
        tot_per = {}
        frac_per = {}

        for chrom, ref_pos in ref_per.items():
            if ref_pos is None or ref_pos.size == 0:
                hit_per[chrom] = 0
                tot_per[chrom] = 0
                frac_per[chrom] = np.nan
                continue

            tot = int(ref_pos.size)
            hit = 0

            test_pos = test_per.get(chrom)
            if test_pos is not None and test_pos.size:
                for p in ref_pos:
                    lo = np.searchsorted(test_pos, p - w_bp, side="left")
                    hi = np.searchsorted(test_pos, p + w_bp, side="right")
                    if hi > lo:
                        hit += 1

            hit_per[chrom] = int(hit)
            tot_per[chrom] = tot
            frac_per[chrom] = (hit / tot) if tot else np.nan

        return hit_per, tot_per, frac_per


    @staticmethod
    def compute_chrom_caller_overlap(
        be_sorted,                       # from extract_breakends(...)
        ref_caller="manta",
        callers=None,
        ignore_callers=("manta_v1.6",),
        svtype="BND",                    # {"BND","DEL","DUP","INV","INS","total"}
        window_bp=100,
        chroms=None,                     # list like ["7"] or ["chr7"] or None
        allowed_samples=None,
        sample_normalizer=None,
        svtypes_all=("BND","DEL","DUP","INV","INS"),
        metric="fraction",               # "fraction" or "count"
        plot=True,
        title=None,
        figsize=(8.5, 6.0),
        show_ref_totals=False,           # optional: annotate ref totals per chrom
    ):
        """
        Returns:
        chrom_list, caller_list, M

        Where M[i,j] is either:
        - fraction (hit/ref_total) if metric="fraction"
        - hit count               if metric="count"
        """
        if metric not in {"fraction", "count"}:
            raise ValueError("metric must be one of {'fraction','count'}")

        # Aggregate across allowed samples only
        agg = SVmethods._aggregate_breakends(
            be_sorted,
            svtypes=set(svtypes_all),
            allowed_samples=allowed_samples,
            sample_normalizer=sample_normalizer
        )
        if ref_caller not in agg:
            raise ValueError(f"Reference caller '{ref_caller}' has no breakends after filtering.")

        def per_chrom_dict(caller):
            if svtype == "total":
                return SVmethods._concat_total_per_chrom(agg.get(caller, {}), svtypes_all)
            return agg.get(caller, {}).get(svtype, {})

        ref_per = per_chrom_dict(ref_caller)

        # Chromosome list: match ref key naming (chr7 vs 7)
        ref_keys = list(ref_per.keys())
        has_chr_prefix = any(str(c).lower().startswith("chr") for c in ref_keys)

        def _norm(c):
            c = str(c).strip()
            base = c[3:] if c.lower().startswith("chr") else c
            if base.upper() in {"M", "MT"}:
                base = "M" if has_chr_prefix else "MT"
            return ("chr" + base) if has_chr_prefix else base

        if chroms is None:
            if "PRIMARY_ORDER" in globals():
                chrom_list = [c for c in PRIMARY_ORDER if c in ref_per]
            else:
                primary = ([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]) if has_chr_prefix \
                        else ([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])
                chrom_list = [c for c in primary if c in ref_per]
            if not chrom_list:  # fallback: whatever exists in ref
                chrom_list = sorted(ref_per.keys())
        else:
            chrom_list = [_norm(c) for c in chroms]

        # Caller list
        all_callers = sorted(be_sorted.keys()) if callers is None else list(callers)
        ignore_set = set(ignore_callers) if ignore_callers else set()
        caller_list = [c for c in all_callers if c not in ignore_set and c != ref_caller and c in agg]

        # Build matrix
        M = np.full((len(chrom_list), len(caller_list)), np.nan, dtype=float)
        ref_totals_vec = np.zeros(len(chrom_list), dtype=int)

        for j, caller in enumerate(caller_list):
            test_per = per_chrom_dict(caller)
            hit_per, tot_per, frac_per = SVmethods._overlap_stats_per_chrom(ref_per, test_per, w_bp=window_bp)

            for i, chrom in enumerate(chrom_list):
                ref_totals_vec[i] = int(tot_per.get(chrom, 0))
                M[i, j] = frac_per.get(chrom, np.nan) if metric == "fraction" else hit_per.get(chrom, np.nan)

        # Plot (optional)
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(M, aspect="auto")

            ax.set_xticks(np.arange(len(caller_list)))
            ax.set_xticklabels(caller_list, rotation=30, ha="right")

            ylabels = chrom_list
            if show_ref_totals:
                ylabels = [f"{c}  (ref={ref_totals_vec[i]})" for i, c in enumerate(chrom_list)]
            ax.set_yticks(np.arange(len(chrom_list)))
            ax.set_yticklabels(ylabels)

            if title is None:
                ylab = "Overlap fraction" if metric == "fraction" else "Overlap count"
                title = f"{ylab} vs {ref_caller} — {svtype} — ±{window_bp} bp"
            ax.set_title(title)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("fraction" if metric == "fraction" else "count")

            plt.tight_layout()
            plt.show()

        return chrom_list, caller_list, M

    @staticmethod
    def plot_chrom_caller_heatmap(chroms, callers, M,
                                svtype="BND", window_bp=100, ref_name="manta",
                                metric="fraction", vmax=None):
        import numpy.ma as ma

        if metric not in {"fraction", "count"}:
            raise ValueError("metric must be one of {'fraction','count'}")

        Mmask = ma.masked_invalid(M)

        fig, ax = plt.subplots(
            figsize=(max(6, 0.9*len(callers)), max(3, 0.35*len(chroms)))
        )

        # Color scaling / labels
        if metric == "fraction":
            vmin = 0.0
            vmax = 1.0 if vmax is None else vmax
            cbar_label = "Fraction of ref breakends matched"
            annot_fmt = "{:.2f}"
            cmap = "Reds"
        else:
            vmin = 0.0
            vmax = (np.nanmax(M) if vmax is None else vmax)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            cbar_label = "Overlapping ref breakends (hit count)"
            annot_fmt = "{:.0f}"
            cmap = "Reds"

        im = ax.imshow(Mmask, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(len(callers)))
        ax.set_yticks(np.arange(len(chroms)))
        ax.set_xticklabels(callers, rotation=40, ha="right")

        # Keep your original row labeling behavior
        ax.set_yticklabels([f"chr{c if c!='MT' else 'M'}" for c in chroms])

        ax.set_title(f"Overlap with {ref_name} (±{window_bp} bp) — {svtype} — {metric}")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

        # Annotate cells
        for i in range(len(chroms)):
            for j in range(len(callers)):
                val = M[i, j]
                if np.isnan(val):
                    continue

                txt = annot_fmt.format(val)

                # pick text color based on relative intensity
                # (works for both fraction and count)
                if metric == "fraction":
                    is_dark = val > 0.5
                else:
                    is_dark = (vmax > 0) and (val / vmax > 0.5)

                ax.text(j, i, txt,
                        ha="center", va="center",
                        color="white" if is_dark else "black",
                        fontsize=_fs(8))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def build_breakend_index_per_sample(
        be_sorted,
        callers,
        svtypes=("BND","DEL","DUP","INV","INS"),
        sample_normalizer=None,
        allowed_samples=None,
    ):
        """
        Returns:
        be_index[sample][chrom] -> sorted np.ndarray of breakend positions
        Union across `callers` and `svtypes`, but kept PER-SAMPLE.
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s
        allowed_set = set(allowed_samples) if allowed_samples is not None else None
        svset = set(svtypes)

        # infer chrom style (chr7 vs 7) from breakends
        want_chr_prefix = None
        for caller in callers:
            sd = be_sorted.get(caller, {})
            for _s, svd in sd.items():
                for _svt, chd in svd.items():
                    for ch in chd.keys():
                        want_chr_prefix = str(ch).lower().startswith("chr")
                        break
                    if want_chr_prefix is not None:
                        break
                if want_chr_prefix is not None:
                    break
            if want_chr_prefix is not None:
                break

        def _norm_chrom(chrom):
            c = str(chrom).strip()
            base = c[3:] if c.lower().startswith("chr") else c
            if base.upper() in {"M","MT"}:
                base = "M" if want_chr_prefix else "MT"
            return ("chr" + base) if want_chr_prefix else base

        tmp = defaultdict(lambda: defaultdict(list))  # sample -> chrom -> list[arr]

        for caller in callers:
            sd = be_sorted.get(caller, {})
            for sample, svdict in sd.items():
                s_norm = sample_normalizer(sample)
                if allowed_set is not None and s_norm not in allowed_set:
                    continue

                for svt, chdict in svdict.items():
                    if svt not in svset:
                        continue
                    for chrom, arr in chdict.items():
                        if arr is None or arr.size == 0:
                            continue
                        ch = _norm_chrom(chrom)
                        tmp[s_norm][ch].append(arr)

        be_index = {}
        for sample, chd in tmp.items():
            be_index[sample] = {}
            for chrom, lst in chd.items():
                cat = np.concatenate(lst) if lst else np.array([], dtype=np.int64)
                if cat.size:
                    cat.sort()
                    be_index[sample][chrom] = cat

        return be_index

    @staticmethod
    def build_be_indices_union_and_percaller(
        be_sorted,
        callers=None,
        svtypes=("BND","DEL","DUP","INV","INS"),
        allowed_samples=None,
        sample_normalizer=None,
        include_total=True,   # add svtype="total" per caller
    ):
        """
        Returns:
        be_index_union[sample][chrom] -> sorted np.ndarray of positions (union of callers+svtypes)
        be_index_by[caller][svtype][sample][chrom] -> sorted np.ndarray of positions
            (and optionally be_index_by[caller]["total"][sample][chrom])
        """
        if sample_normalizer is None:
            sample_normalizer = lambda s: s
        allowed_set = set(allowed_samples) if allowed_samples is not None else None

        callers_use = list(be_sorted.keys()) if callers is None else list(callers)

        # Infer whether breakends use 'chr' prefix from first non-empty chrom we see
        want_chr_prefix = None
        for c in callers_use:
            sd = be_sorted.get(c, {})
            for _s, svd in sd.items():
                for _svt, chd in svd.items():
                    for ch in chd.keys():
                        want_chr_prefix = str(ch).lower().startswith("chr")
                        break
                    if want_chr_prefix is not None:
                        break
                if want_chr_prefix is not None:
                    break
            if want_chr_prefix is not None:
                break

        be_index_union = defaultdict(lambda: defaultdict(list))
        be_index_by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for caller in callers_use:
            sd = be_sorted.get(caller, {})
            for sample, svdict in sd.items():
                s_norm = sample_normalizer(sample)
                if allowed_set is not None and s_norm not in allowed_set:
                    continue

                for svt, chdict in svdict.items():
                    if svt not in set(svtypes):
                        continue

                    for chrom, arr in chdict.items():
                        if arr is None or arr.size == 0:
                            continue

                        ch = SNVmethods._norm_chrom(chrom, want_chr_prefix=want_chr_prefix)

                        be_index_by[caller][svt][s_norm][ch].append(arr)
                        be_index_union[s_norm][ch].append(arr)

        # finalize: concat + sort
        def _finalize(d):
            out = {}
            for sample, chd in d.items():
                out[sample] = {}
                for chrom, lst in chd.items():
                    cat = np.concatenate(lst) if lst else np.array([], dtype=np.int64)
                    if cat.size:
                        cat.sort()
                        out[sample][chrom] = cat
            return out

        be_index_union = _finalize(be_index_union)

        # per-caller per-svtype finalize
        be_index_by_final = {}
        for caller, svtd in be_index_by.items():
            be_index_by_final[caller] = {}
            for svt, sd in svtd.items():
                be_index_by_final[caller][svt] = _finalize(sd)

            if include_total:
                # total = concat across svtypes per caller
                total_sd = defaultdict(lambda: defaultdict(list))
                for svt in svtypes:
                    sd = be_index_by_final[caller].get(svt, {})
                    for sample, chd in sd.items():
                        for chrom, arr in chd.items():
                            if arr is not None and arr.size:
                                total_sd[sample][chrom].append(arr)
                be_index_by_final[caller]["total"] = _finalize(total_sd)

        return be_index_union, be_index_by_final

    @staticmethod
    def nearest_distance_to_breakends(pos: int, breaks):
        """
        breaks: sorted np.ndarray of positions
        returns min absolute distance (int) or np.nan if breaks empty
        """
        if breaks is None or breaks.size == 0:
            return np.nan
        i = np.searchsorted(breaks, pos)
        best = None
        if i < breaks.size:
            best = abs(int(breaks[i]) - pos)
        if i > 0:
            d2 = abs(int(breaks[i-1]) - pos)
            best = d2 if best is None else min(best, d2)
        return best if best is not None else np.nan


#############################################
#### METHODS RELATED TO SNV CALLERS ONLY ####
#############################################
class SNVmethods:
    @staticmethod
    def get_smrest_vcfs(root_dir: str, sample_id: str) -> list[str]:
        """
        Return list of smrest VCFs for a sample:
        $root_dir/$sample_id/${sample_id}_chr${chr}_0.6_calls.vcf
        """
        sample_dir = os.path.join(root_dir, sample_id)
        vcfs = []

        for chrom in CHROMS:
            fname = f"{sample_id}_chr{chrom}_0.6_calls.vcf"
            path = os.path.join(sample_dir, fname)
            if os.path.exists(path):
                vcfs.append(path)

        if not vcfs:
            # Optional fallback: any calls.vcf under the sample dir
            fallback = sorted(glob.glob(os.path.join(sample_dir, "*calls.vcf*")))
            if fallback:
                print(f"[WARN] No explicit chr files for {sample_id}; using fallback(s):")
                for p in fallback:
                    print("   ", p)
                vcfs = fallback

        if not vcfs:
            raise FileNotFoundError(f"No smrest VCFs found for {sample_id} in {sample_dir}")

        return vcfs
    
    @staticmethod
    def get_clair3_vcfs(root_dir: str, sample_id: str) -> str:
        sample_dir = os.path.join(root_dir, sample_id)
        path = os.path.join(sample_dir, "merge_output.vcf.gz")
        if os.path.exists(path):
            return path

        # Fallback: prefer anything with "merge" in its name
        vcfs = sorted(glob.glob(os.path.join(sample_dir, "*.vcf.gz")))
        preferred = [v for v in vcfs if "merge" in os.path.basename(v)]
        if preferred:
            if len(preferred) > 1:
                print(f"[WARN] Multiple 'merge' VCFs for {sample_id}; using {preferred[0]}")
            return preferred[0]

        raise FileNotFoundError(f"No Clair3 merge_output VCF found for {sample_id} in {sample_dir}")

    @staticmethod
    def get_clairs_to_vcfs(root_dir: str, sample_id: str, include_indel: bool = True) -> list[str]:
        sample_dir = os.path.join(root_dir, sample_id)
        
        # Always include SNV
        candidates = [os.path.join(sample_dir, "snv.vcf.gz")]
        
        # Optionally include indel
        if include_indel:
            indel_path = os.path.join(sample_dir, "indel.vcf.gz")
            if os.path.exists(indel_path):
                candidates.append(indel_path)
            else:
                print(f"Indel file is absent for {sample_id}, continuing without it")
        else:
            print("Not including ClairS-TO INDELs")
        
        vcfs = [p for p in candidates if os.path.exists(p)]
        
        if not vcfs:
            # Fallback: any *.vcf.gz
            vcfs = sorted(glob.glob(os.path.join(sample_dir, "*.vcf.gz")))
        
        if not vcfs:
            raise FileNotFoundError(f"No ClairS-TO VCFs found for {sample_id} in {sample_dir}")
        
        return vcfs
    
    @staticmethod
    def get_clairs_vcfs(root_dir: str, sample_id: str) -> str:
        sample_dir = os.path.join(root_dir, sample_id)
        candidate = os.path.join(sample_dir, "output.vcf.gz")
        if os.path.exists(candidate):
            return candidate
        # Fallback: any *.vcf.gz
        vcfs = sorted(glob.glob(os.path.join(sample_dir, "*.vcf.gz")))
        if not vcfs:
            raise FileNotFoundError(f"No ClairS VCFs found for {sample_id} in {sample_dir}")
        return vcfs[0]

    @staticmethod
    def get_pmdv_vcfs(pmdv_root, sample):
        vcf_pattern = os.path.join(pmdv_root, sample, "*.vcf.gz")
        files = [f for f in glob.glob(vcf_pattern) if not f.endswith(".tbi")]
        return files[0] if files else None

    @staticmethod
    def _extract_vaf_and_dp(call) -> tuple[float, float]:
        """
        Best-effort VAF/DP extraction from a vcfpy Call object.
        Tries AF, then AD/DP, then VD/DP style fields.
        Returns (vaf, dp) with np.nan if unavailable.
        """
        data = getattr(call, "data", {}) or {}

        # DP
        dp = data.get("DP")
        if isinstance(dp, (list, tuple)):
            dp = dp[0] if dp else None
        try:
            dp = int(dp)
        except Exception:
            dp = np.nan

        vaf = np.nan

        # 1) AF (preferred if present)
        af = data.get("AF")
        if af is not None:
            if isinstance(af, (list, tuple)):
                af = af[0] if af else None
            try:
                vaf = float(af)
                return vaf, dp
            except Exception:
                pass

        # 2) AD + DP
        ad = data.get("AD")
        if ad is not None:
            # AD often like [ref, alt1, alt2, ...]
            if isinstance(ad, (list, tuple)) and len(ad) >= 2:
                ref_ad, alt_ad = ad[0], ad[1]
            else:
                # If only alt depth stored, treat it as alt
                ref_ad = None
                alt_ad = ad[0] if isinstance(ad, (list, tuple)) else ad
            try:
                alt_ad = float(alt_ad)
                if dp and dp > 0:
                    vaf = alt_ad / dp
                    return vaf, dp
            except Exception:
                pass

        # 3) VD + DP (sometimes used for variant depth)
        vd = data.get("VD")
        if vd is not None:
            if isinstance(vd, (list, tuple)):
                vd = vd[0]
            try:
                alt_depth = float(vd)
                if dp and dp > 0:
                    vaf = alt_depth / dp
            except Exception:
                pass

        return vaf, dp
    
    @staticmethod
    def plot_vaf_comparison_by_sample(
        clairs_all: pd.DataFrame,
        clairsto_all: pd.DataFrame,
        smrest_all: pd.DataFrame | None = None,
        mutect_all: pd.DataFrame | None = None,
    ):
        """
        Compare VAF distributions between ClairS and ClairS-TO per sample.

        Parameters
        ----------
        clairs_all    : pd.DataFrame  output of compute_longread_snv_overlap
        clairsto_all  : pd.DataFrame  output of compute_longread_snv_overlap
        smrest_all    : pd.DataFrame  optional
        mutect_all    : pd.DataFrame  optional
        """
        clairs_all = clairs_all if clairs_all is not None else pd.DataFrame()
        clairsto_all = clairsto_all if clairsto_all is not None else pd.DataFrame()
        smrest_all = smrest_all if smrest_all is not None else pd.DataFrame()
        mutect_all = mutect_all if mutect_all is not None else pd.DataFrame()
        
        if clairs_all.empty and clairsto_all.empty:
            print("[INFO] Both ClairS and ClairS-TO are empty; skipping.")
            return

        samples = sorted(set(
            list(clairs_all["sample"].unique() if not clairs_all.empty else []) +
            list(clairsto_all["sample"].unique() if not clairsto_all.empty else [])
        ))

        n = len(samples)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, s in zip(axes, samples):
            data = []
            labels = []

            # --- ClairS ---
            if not clairs_all.empty:
                df_cs = clairs_all[clairs_all["sample"] == s].dropna(subset=["vaf"])
                if not df_cs.empty:
                    data.append(df_cs["vaf"].values)
                    labels.append("ClairS")

                    shared_c3 = df_cs.loc[df_cs["overlap_category"] == "Shared_with_Clair3", "vaf"].values
                    if shared_c3.size > 0:
                        data.append(shared_c3)
                        labels.append("ClairS ∩ Clair3")

            # --- ClairS-TO ---
            if not clairsto_all.empty:
                df_csto = clairsto_all[clairsto_all["sample"] == s].dropna(subset=["vaf"])
                if not df_csto.empty:
                    data.append(df_csto["vaf"].values)
                    labels.append("ClairS-TO")

                    shared_c3 = df_csto.loc[df_csto["overlap_category"] == "Shared_with_Clair3", "vaf"].values
                    if shared_c3.size > 0:
                        data.append(shared_c3)
                        labels.append("ClairS-TO ∩ Clair3")

            # --- Shared ClairS ∩ ClairS-TO ---
            if not clairs_all.empty and not clairsto_all.empty:
                df_cs = clairs_all[clairs_all["sample"] == s].dropna(subset=["vaf"])
                df_csto = clairsto_all[clairsto_all["sample"] == s].dropna(subset=["vaf"])
                if "key" in df_cs.columns and "key" in df_csto.columns:
                    shared_keys = set(df_cs["key"]) & set(df_csto["key"])
                    if shared_keys:
                        vals = df_cs.loc[df_cs["key"].isin(shared_keys), "vaf"].values
                        if vals.size > 0:
                            data.append(vals)
                            labels.append("ClairS ∩ ClairS-TO")

            # --- smrest ---
            if not smrest_all.empty and "vaf" in smrest_all.columns:
                df_sm = smrest_all[smrest_all["sample"] == s].dropna(subset=["vaf"])
                if not df_sm.empty:
                    data.append(df_sm["vaf"].values)
                    labels.append("smrest")

            # --- mutect ---
            if not mutect_all.empty and "vaf" in mutect_all.columns:
                df_mut = mutect_all[mutect_all["sample"] == s].dropna(subset=["vaf"])
                if not df_mut.empty:
                    data.append(df_mut["vaf"].values)
                    labels.append("Mutect2")

            if data:
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.set_title(s)
                ax.set_ylabel("VAF")
                ax.tick_params(axis="x", rotation=30)
            else:
                ax.set_title(f"{s}\n(no data)")
                ax.set_visible(True)

        fig.suptitle("VAF distribution by caller", y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_small_variants(vcf_path: str,
                            sample_label: str,
                            source: str) -> pd.DataFrame:
        """
        Load PASS small variants from a VCF (plain or gzipped) into a DataFrame.

        Supports both .vcf and .vcf.gz files.
        """
        if not os.path.exists(vcf_path):
            print(f"[WARN] File does not exist: {vcf_path}")
            return pd.DataFrame()

        # For .gz files, open with gzip in text mode
        if vcf_path.endswith(".gz"):
            reader = vcfpy.Reader.from_stream(gzip.open(vcf_path, "rt"))
        else:
            reader = vcfpy.Reader.from_path(vcf_path)

        records = []

        for rec in reader:
            if not VariationAnalyzer.vcf_record_is_pass(rec):
                continue

            chrom = SVmethods._normalize_chrom(rec.CHROM)
            pos = int(rec.POS)
            ref = rec.REF

            if not rec.ALT:
                continue
            alt_obj = rec.ALT[0]
            alt = getattr(alt_obj, "value", str(alt_obj))

            key = (chrom, pos, ref, alt)

            if not rec.calls:
                continue
            call = rec.calls[0]
            vaf, dp = SNVmethods._extract_vaf_and_dp(call)

            filters = rec.FILTER
            filt_str = "PASS" if not filters else ";".join(filters)

            records.append({
                "sample": sample_label,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "key": key,
                "source": source,
                "filter": filt_str,
                "vaf": vaf,
                "dp": dp,
            })

        return pd.DataFrame.from_records(records)

    @staticmethod
    def load_small_variants_region(
        vcf_path: str,
        sample_label: str,
        source: str,
        chrom: str,
        start: int,
        end: int,
    ) -> pd.DataFrame:
        """
        Load PASS small variants from a tabix-indexed VCF restricted to chrom:start-end.
        Uses vcfpy.fetch() — requires a .tbi index next to the VCF.
        chrom should be bare (e.g. "6"); chr-prefix is added automatically.
        Falls back to full-file scan if the index is absent.
        """
        if not os.path.exists(vcf_path):
            print(f"[WARN] File does not exist: {vcf_path}")
            return pd.DataFrame()

        has_index = os.path.exists(vcf_path + ".tbi")
        if vcf_path.endswith(".gz"):
            reader = vcfpy.Reader.from_stream(gzip.open(vcf_path, "rt")) if not has_index \
                     else vcfpy.Reader.from_path(vcf_path)
        else:
            reader = vcfpy.Reader.from_path(vcf_path)

        # Use tabix region fetch when index is present
        chr_chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
        if has_index:
            try:
                record_iter = reader.fetch(chr_chrom, start, end)
            except Exception:
                record_iter = iter(reader)
        else:
            record_iter = iter(reader)

        records = []
        for rec in record_iter:
            if not VariationAnalyzer.vcf_record_is_pass(rec):
                continue
            rec_chrom = SVmethods._normalize_chrom(rec.CHROM)
            rec_pos   = int(rec.POS)
            if not has_index:
                if rec_chrom != chrom or not (start <= rec_pos <= end):
                    continue
            ref = rec.REF
            if not rec.ALT:
                continue
            alt_obj = rec.ALT[0]
            alt = getattr(alt_obj, "value", str(alt_obj))
            key = (rec_chrom, rec_pos, ref, alt)
            if not rec.calls:
                continue
            vaf, dp = SNVmethods._extract_vaf_and_dp(rec.calls[0])
            filters   = rec.FILTER
            filt_str  = "PASS" if not filters else ";".join(filters)
            records.append({
                "sample": sample_label,
                "chrom":  rec_chrom,
                "pos":    rec_pos,
                "ref":    ref,
                "alt":    alt,
                "key":    key,
                "source": source,
                "filter": filt_str,
                "vaf":    vaf,
                "dp":     dp,
            })

        return pd.DataFrame.from_records(records)

    @staticmethod
    def load_clairs_for_gene(
        clairsto_root: str,
        samples: list,
        gene: "str | dict",
        flank_bp: int = 200_000,
        include_indel: bool = True,
        min_vaf: float = 0.1,
        max_vaf: float = 0.6,
    ) -> pd.DataFrame:
        """
        Load ClairS-TO variants for all samples restricted to gene ± flank_bp.
        Fast: uses tabix index so only the region is read from each VCF.
        VAF filter: 0.1 <= VAF <= 0.6 by default (somatic range, excludes germline).
        gene: key from GENE_COORDS (e.g. "TBXT") or dict with chrom/start/end.
        """
        if isinstance(gene, str):
            coords    = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords    = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        chrom     = coords["chrom"]
        win_start = coords["start"] - flank_bp
        win_end   = coords["end"]   + flank_bp

        frames = []
        for sample in samples:
            try:
                vcfs = SNVmethods.get_clairs_to_vcfs(clairsto_root, sample, include_indel=include_indel)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue
            for vf in vcfs:
                df = SNVmethods.load_small_variants_region(
                    vf, sample, "clairs-to", chrom, win_start, win_end
                )
                if not df.empty:
                    frames.append(df)

        if not frames:
            print(f"[WARN] No ClairS-TO variants loaded for {gene_name}.")
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if "vaf" in result.columns and (min_vaf is not None or max_vaf is not None):
            before = len(result)
            vaf = result["vaf"].fillna(0)
            mask = pd.Series(True, index=result.index)
            if min_vaf is not None:
                mask &= (vaf >= min_vaf)
            if max_vaf is not None:
                mask &= (vaf <= max_vaf)
            result = result.loc[mask]
            print(f"Loaded {len(result):,} ClairS-TO variants in {gene_name} window "
                  f"(VAF {min_vaf}–{max_vaf}, {before - len(result):,} filtered out)")
        else:
            print(f"Loaded {len(result):,} ClairS-TO variants in {gene_name} window across {len(samples)} samples")
        return result

    @staticmethod
    def load_smrest_variants(vcf_path: str, sample_label: str) -> pd.DataFrame:
        """
        Load smrest site-only calls from a VCF into a DataFrame.

        Columns:
        sample, chrom, pos, ref, alt, key, source, filter, vaf, dp

        VAF/DP are derived from INFO:
        - HaplotypeVAF: list of floats; we take the max as 'vaf'
        - HaplotypeDepth: list of ints; we take the sum as 'dp'
        """
        reader = vcfpy.Reader.from_path(vcf_path)
        records = []

        for rec in reader:
            # --- PASS filter ---
            if not VariationAnalyzer.vcf_record_is_pass(rec):
                continue

            chrom = SVmethods._normalize_chrom(rec.CHROM)
            pos = int(rec.POS)
            ref = rec.REF

            if not rec.ALT:
                continue
            alt_obj = rec.ALT[0]
            alt = getattr(alt_obj, "value", str(alt_obj))

            key = (chrom, pos, ref, alt)

            # FILTER string for bookkeeping only
            filters = rec.FILTER
            if not filters:
                filt_str = "PASS"
            else:
                filt_str = ";".join(filters)

            info = rec.INFO or {}

            # VAF from HaplotypeVAF
            vaf = np.nan
            hvaf = info.get("HaplotypeVAF")
            if hvaf is not None:
                try:
                    if isinstance(hvaf, (list, tuple)):
                        vals = [float(x) for x in hvaf if x is not None]
                        if vals:
                            vaf = max(vals)
                    else:
                        vaf = float(hvaf)
                except Exception:
                    vaf = np.nan

            # DP from HaplotypeDepth
            dp = np.nan
            hdp = info.get("HaplotypeDepth")
            if hdp is not None:
                try:
                    if isinstance(hdp, (list, tuple)):
                        vals = [int(x) for x in hdp if x is not None]
                        if vals:
                            dp = sum(vals)
                    else:
                        dp = int(hdp)
                except Exception:
                    dp = np.nan

            records.append({
                "sample": sample_label,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "key": key,
                "source": "smrest",
                "filter": filt_str,
                "vaf": vaf,
                "dp": dp,
            })

        return pd.DataFrame.from_records(records)

    @staticmethod
    def load_pmdv_variants(vcf_path: str, sample_label: str, source: str = "pmdv") -> pd.DataFrame:
        import gzip
        print(f"[DEBUG] vcf_path={vcf_path!r}, endswith_gz={str(vcf_path).endswith('.gz')}")
        # Detect gzip by magic bytes, not filename
        def _open(path):
            with open(path, "rb") as f:
                magic = f.read(2)
            if magic == b"\x1f\x8b":
                return gzip.open(path, "rt", encoding="utf-8")
            else:
                return open(path, "r", encoding="utf-8")

        records = []
        try:
            with _open(vcf_path) as f:
                for line in f:
                    if line.startswith("##"):
                        continue
                    if line.startswith("#CHROM"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) < 8:
                        continue

                    chrom, pos, id_, ref, alt_str, qual, filt, info = parts[:8]
                    format_str = parts[8] if len(parts) > 8 else ""
                    call_str   = parts[9] if len(parts) > 9 else ""

                    if filt not in ("PASS", "."):
                        continue

                    chrom = SVmethods._normalize_chrom(chrom)
                    pos   = int(pos)
                    alt   = alt_str.split(",")[0]
                    key   = (chrom, pos, ref, alt)

                    vaf, dp = None, None
                    if format_str and call_str:
                        fmt = dict(zip(format_str.split(":"), call_str.split(":")))
                        if "VAF" in fmt:
                            try:
                                vaf = float(fmt["VAF"].split(",")[0])
                            except (ValueError, TypeError):
                                pass
                        if vaf is None and "AD" in fmt:
                            try:
                                ad = [int(x) for x in fmt["AD"].split(",")]
                                total = sum(ad)
                                if total > 0:
                                    vaf = ad[1] / total
                            except (ValueError, TypeError, IndexError):
                                pass
                        if "DP" in fmt:
                            try:
                                dp = int(fmt["DP"])
                            except (ValueError, TypeError):
                                pass

                    records.append({
                        "sample": sample_label,
                        "chrom":  chrom,
                        "pos":    pos,
                        "ref":    ref,
                        "alt":    alt,
                        "key":    key,
                        "source": source,
                        "filter": filt,
                        "vaf":    vaf,
                        "dp":     dp,
                    })

        except Exception as e:
            print(f"[WARN] Failed to load PMDV for {sample_label}: {e}")
            return pd.DataFrame()

        return pd.DataFrame.from_records(records)
    
    @staticmethod
    # ----------------------
    # Helper function: extract SNV keys from a VCF
    # ----------------------
    def load_variants(vcf_path, source=None, include_indels=False, use_cache=True):
        """Return both a set of keys and a DataFrame for the VCF"""
        if vcf_path is None or not Path(vcf_path).exists():
            return set(), pd.DataFrame()
        
        cache_path_keys = Path(str(vcf_path) + f".keys.{int(include_indels)}.pkl")
        cache_path_df = Path(str(vcf_path) + f".df.{int(include_indels)}.pkl")

        # Try cache
        if use_cache and cache_path_keys.exists() and cache_path_df.exists():
            try:
                with open(cache_path_keys, "rb") as f:
                    keys = pickle.load(f)
                with open(cache_path_df, "rb") as f:
                    df = pickle.load(f)
                return keys, df
            except Exception:
                pass

        keys = set()
        rows = []
        with gzip.open(vcf_path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                chrom, pos, id_, ref, alt, qual, filt, info, *rest = line.strip().split("\t")
                if not include_indels and (len(ref) != 1 or len(alt) != 1):
                    continue
                key = f"{chrom}_{pos}_{ref}_{alt}"
                keys.add(key)
                row = {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt, "key": key}
                if source:
                    row["source"] = source
                rows.append(row)
        df = pd.DataFrame(rows)

        if use_cache:
            try:
                with open(cache_path_keys, "wb") as f:
                    pickle.dump(keys, f)
                with open(cache_path_df, "wb") as f:
                    pickle.dump(df, f)
            except Exception:
                pass

        return keys, df

    @staticmethod
    # variation_analyzer.py (partial, SNVmethods)
    # ----------------------------
    # Top-level worker for multiprocessing
    # ----------------------------
    def _process_sample(sample, smrest_root, clair3_root, clairs_to_root, clairs_roots, pmdv_root, include_indels):
        """
        Load SNVs for a single sample across multiple callers.
        """
        dfs = {}
        row = {"sample": sample}

        # --- PMDV (fixed) ---
        if pmdv_root:
            pmdv_vcf = SNVmethods.get_pmdv_vcf(pmdv_root, sample)
            df_pmdv = pd.DataFrame()
            # Guard: reject .tbi or anything that isn't a VCF/VCF.gz
            if pmdv_vcf and str(pmdv_vcf).endswith(".tbi"):
                print(f"[WARN] get_pmdv_vcf returned a .tbi file for {sample}: {pmdv_vcf}")
                pmdv_vcf = None
            if pmdv_vcf and os.path.exists(pmdv_vcf):
                try:
                    # Pass path string, let load_pmdv_variants handle .gz
                    df_pmdv = SNVmethods.load_pmdv_variants(pmdv_vcf, sample, source="pmdv")
                    if not include_indels and "type" in df_pmdv.columns:
                        df_pmdv = df_pmdv[df_pmdv["type"]=="SNV"]
                except Exception as e:
                    print(f"[WARN] Failed to load PMDV for {sample}: {e}")
            else:
                print(f"[WARN] PMDV VCF not found for {sample}: {pmdv_vcf}")
            dfs["pmdv"] = df_pmdv
            row["pmdv_total"] = len(df_pmdv)

        # --- SMREST ---
        if smrest_root:
            sm_vcfs = SNVmethods.get_smrest_vcfs(smrest_root, sample)
            df_sm = pd.concat([SNVmethods.load_smrest_variants(v, sample) for v in sm_vcfs], ignore_index=True) if sm_vcfs else pd.DataFrame()
            if not include_indels and "type" in df_sm.columns:
                df_sm = df_sm[df_sm["type"]=="SNV"]
            dfs["smrest"] = df_sm
            row["smrest_total"] = len(df_sm)

        # --- Clair3 ---
        if clair3_root:
            c3_vcf = SNVmethods.get_clair3_vcf(clair3_root, sample)
            df_c3 = SNVmethods.load_small_variants(c3_vcf, sample, source="clair3")
            if not include_indels and "type" in df_c3.columns:
                df_c3 = df_c3[df_c3["type"]=="SNV"]
            dfs["clair3"] = df_c3
            row["clair3_total"] = len(df_c3)

        # --- ClairS-TO ---
        if clairs_to_root:
            csto_vcfs = SNVmethods.get_clairs_to_vcfs(clairs_to_root, sample, include_indel=include_indels)
            df_csto = pd.concat([SNVmethods.load_small_variants(v, sample, source="clairs_to") for v in csto_vcfs], ignore_index=True) if csto_vcfs else pd.DataFrame()
            dfs["clairs_to"] = df_csto
            row["clairs_to_total"] = len(df_csto)

        # --- multiple ClairS versions ---
        if clairs_roots:
            for name, root in clairs_roots.items():
                cs_vcf = SNVmethods.get_clairs_vcfs(root, sample)
                df_cs = SNVmethods.load_small_variants(cs_vcf, sample, source=name)
                if not include_indels and "type" in df_cs.columns:
                    df_cs = df_cs[df_cs["type"]=="SNV"]
                dfs[name] = df_cs
                row[f"{name}_total"] = len(df_cs)

        return row, dfs

    @staticmethod
    # ----------------------------
    # Fast multi-sample SNV overlap
    # ----------------------------
    def compute_longread_snv_overlap_fast(
        samples,
        smrest_root=None,
        clair3_root=None,
        clairs_to_root=None,
        clairs_roots=None,
        pmdv_root=None,
        include_indels=False,
        n_jobs=4,
        use_cache=True
    ):
        """
        Compute per-sample SNV overlaps across multiple long-read callers (Clair3, ClairS variants, SMREST, PMDV)
        Returns: overlap_summary, clairsto_all_raw, clairs_all_raw, clairs_all_marc_raw, smrest_all_raw, clair3_all_raw, pmdv_all_raw
        """
        all_rows = []
        caller_keys = ["clairs_to", "smrest", "clair3", "pmdv"]
        if clairs_roots:
            caller_keys += list(clairs_roots.keys())
        all_dfs = {k: [] for k in caller_keys}

        # --- Parallel processing ---
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            results = list(
                exe.map(
                    SNVmethods._process_sample,
                    samples,
                    [smrest_root]*len(samples),
                    [clair3_root]*len(samples),
                    [clairs_to_root]*len(samples),
                    [clairs_roots]*len(samples),
                    [pmdv_root]*len(samples),
                    [include_indels]*len(samples)
                )
            )

        # --- Aggregate ---
        for row, dfs in results:
            all_rows.append(row)
            for k, df in dfs.items():
                if not df.empty:
                    all_dfs[k].append(df)

        overlap_summary = pd.DataFrame(all_rows).set_index("sample").sort_index()

        clairsto_all_raw      = pd.concat(all_dfs.get("clairs_to", []), ignore_index=True) if all_dfs.get("clairs_to") else pd.DataFrame()
        clairs_all_raw        = pd.concat(all_dfs.get("clairs", []), ignore_index=True) if all_dfs.get("clairs") else pd.DataFrame()
        clairs_all_marc_raw   = pd.concat(all_dfs.get("clairs_marc", []), ignore_index=True) if all_dfs.get("clairs_marc") else pd.DataFrame()
        smrest_all_raw        = pd.concat(all_dfs.get("smrest", []), ignore_index=True) if all_dfs.get("smrest") else pd.DataFrame()
        clair3_all_raw        = pd.concat(all_dfs.get("clair3", []), ignore_index=True) if all_dfs.get("clair3") else pd.DataFrame()
        pmdv_all_raw          = pd.concat(all_dfs.get("pmdv", []), ignore_index=True) if all_dfs.get("pmdv") else pd.DataFrame()

        return overlap_summary, clairsto_all_raw, clairs_all_raw, clairs_all_marc_raw, smrest_all_raw, clair3_all_raw, pmdv_all_raw

    @staticmethod
    def compute_longread_snv_overlap(
        samples,
        smrest_root: str | None = None,
        clair3_root: str | None = None,
        clairs_to_root: str | None = None,
        clairs_root: str | None = None,
    ):
        """
        Compute per-sample SNV overlaps across available long-read callers.

        Supported callers:
        - smrest       (per-chrom VCFs)
        - Clair3       (merged VCF)
        - ClairS-TO    (snv + indel VCFs)
        - ClairS       (snv + indel VCFs)

        Missing callers are ignored gracefully.

        Returns
        -------
        overlap_summary : pd.DataFrame
        clairsto_all    : pd.DataFrame
        clairs_all      : pd.DataFrame
        smrest_all      : pd.DataFrame
        clair3_all      : pd.DataFrame
        """
        overlap_rows = []
        clairs_parts = []
        clairsto_parts = []
        smrest_parts = []
        clair3_parts = []

        for s in samples:
            print(f"Processing sample {s}")

            # ------------------
            # smrest
            # ------------------
            set_sm = set()
            df_sm = pd.DataFrame()

            if smrest_root is not None:
                try:
                    sm_vcfs = SNVmethods.get_smrest_vcfs(smrest_root, s)
                    sm_parts = [
                        SNVmethods.load_smrest_variants(vpath, s)
                        for vpath in sm_vcfs
                    ]
                    if sm_parts:
                        df_sm = pd.concat(sm_parts, ignore_index=True)
                        if "key" in df_sm.columns:
                            set_sm = set(df_sm["key"])
                            smrest_parts.append(df_sm)
                except FileNotFoundError:
                    pass

            # ------------------
            # Clair3
            # ------------------
            set_c3 = set()
            df_c3 = pd.DataFrame()

            if clair3_root is not None:
                try:
                    c3_vcf = SNVmethods.get_clair3_vcf(clair3_root, s)
                    df_c3 = SNVmethods.load_small_variants(c3_vcf, s, source="clair3")
                    if not df_c3.empty and "key" in df_c3.columns:
                        set_c3 = set(df_c3["key"])
                        clair3_parts.append(df_c3)
                except FileNotFoundError:
                    pass

            # ------------------
            # ClairS-TO
            # ------------------
            set_csto = set()
            df_csto = pd.DataFrame()

            if clairs_to_root is not None:
                try:
                    csto_vcfs = SNVmethods.get_clairs_to_vcfs(clairs_to_root, s, include_indel=False)
                    csto_parts = [
                        SNVmethods.load_small_variants(vpath, s, source="clairs_to")
                        for vpath in csto_vcfs
                    ]
                    if csto_parts:
                        df_csto = pd.concat(csto_parts, ignore_index=True)
                        if "key" in df_csto.columns:
                            set_csto = set(df_csto["key"])
                except FileNotFoundError:
                    pass

            # ------------------
            # ClairS
            # ------------------
            set_cs = set()
            df_cs = pd.DataFrame()

            if clairs_root is not None:
                try:
                    cs_vcf = SNVmethods.get_clairs_vcfs(clairs_root, s)  # returns path to output.vcf.gz
                    df_cs = SNVmethods.load_small_variants(cs_vcf, s, source="clairs")
                    if not df_cs.empty and "key" in df_cs.columns:
                        set_cs = set(df_cs["key"])
                except FileNotFoundError:
                    pass

            # ------------------
            # Overlaps (safe with empty sets)
            # ------------------
            shared_c3_csto  = set_c3 & set_csto
            shared_c3_cs    = set_c3 & set_cs
            shared_c3_sm    = set_c3 & set_sm
            shared_csto_cs  = set_csto & set_cs
            shared_csto_sm  = set_csto & set_sm
            shared_cs_sm    = set_cs & set_sm
            shared_all4     = set_c3 & set_csto & set_cs & set_sm

            overlap_rows.append({
                "sample": s,

                # totals
                "clair3_total":    len(set_c3),
                "clairs_to_total": len(set_csto),
                "clairs_total":    len(set_cs),
                "smrest_total":    len(set_sm),

                # unique
                "clair3_only":    len(set_c3   - set_csto - set_cs - set_sm),
                "clairs_to_only": len(set_csto - set_c3   - set_cs - set_sm),
                "clairs_only":    len(set_cs   - set_c3   - set_csto - set_sm),
                "smrest_only":    len(set_sm   - set_c3   - set_csto - set_cs),

                # pairwise
                "shared_c3_clairs_to":     len(shared_c3_csto),
                "shared_c3_clairs":        len(shared_c3_cs),
                "shared_c3_smrest":        len(shared_c3_sm),
                "shared_clairs_to_clairs": len(shared_csto_cs),
                "shared_clairs_to_smrest": len(shared_csto_sm),
                "shared_clairs_smrest":    len(shared_cs_sm),

                # quad
                "shared_all_four": len(shared_all4),
            })

            # ------------------
            # Label ClairS-TO
            # ------------------
            if not df_csto.empty:
                df_csto = df_csto.copy()
                if set_c3:
                    df_csto["overlap_category"] = np.where(
                        df_csto["key"].isin(shared_c3_csto),
                        "Shared_with_Clair3",
                        "ClairS_TO_only"
                    )
                else:
                    df_csto["overlap_category"] = "ClairS_TO_only"
                clairsto_parts.append(df_csto)

            # ------------------
            # Label ClairS
            # ------------------
            if not df_cs.empty:
                df_cs = df_cs.copy()
                if set_c3:
                    df_cs["overlap_category"] = np.where(
                        df_cs["key"].isin(shared_c3_cs),
                        "Shared_with_Clair3",
                        "ClairS_only"
                    )
                else:
                    df_cs["overlap_category"] = "ClairS_only"
                clairs_parts.append(df_cs)

        # ------------------
        # Wrap-up
        # ------------------
        overlap_summary = (
            pd.DataFrame(overlap_rows)
            .set_index("sample")
            .sort_index()
            if overlap_rows else
            pd.DataFrame()
        )

        clairsto_all = pd.concat(clairsto_parts, ignore_index=True) if clairsto_parts else pd.DataFrame()
        clairs_all   = pd.concat(clairs_parts,   ignore_index=True) if clairs_parts   else pd.DataFrame()
        smrest_all   = pd.concat(smrest_parts,   ignore_index=True) if smrest_parts   else pd.DataFrame()
        clair3_all   = pd.concat(clair3_parts,   ignore_index=True) if clair3_parts   else pd.DataFrame()

        return overlap_summary, clairsto_all, clairs_all, smrest_all, clair3_all

    @staticmethod
    def load_caller_vcfs(samples, caller, root, include_indels=False, n_jobs=4):
        """
        Minimal parallel loader for a single caller.
        caller: one of 'clairs_to', 'clairs', 'smrest', 'clair3', 'pmdv'
        Returns: concatenated DataFrame
        """
        loader_map = {
            "clairs_to": (lambda s: SNVmethods.get_clairs_to_vcfs(root, s, include_indel=include_indels),
                        lambda p, s: SNVmethods.load_small_variants(p, s, source="clairs_to")),
            "clairs":    (lambda s: SNVmethods.get_clairs_vcfs(root, s),
                        lambda p, s: SNVmethods.load_small_variants(p, s, source="clairs")),
            "smrest":    (lambda s: SNVmethods.get_smrest_vcfs(root, s),
                        lambda p, s: SNVmethods.load_smrest_variants(p, s)),
            "clair3":    (lambda s: SNVmethods.get_clair3_vcfs(root, s),
                        lambda p, s: SNVmethods.load_small_variants(p, s, source="clair3")),
            "pmdv":      (lambda s: SNVmethods.get_pmdv_vcfs(root, s),
                        lambda p, s: SNVmethods.load_pmdv_variants(p, s, source="pmdv")),
        }
        if caller not in loader_map:
            raise ValueError(f"Unknown caller '{caller}'. Choose from: {list(loader_map)}")

        getter, parser = loader_map[caller]

        def _load(s):
            paths = getter(s)
            if not isinstance(paths, list):
                paths = [paths]
            dfs = [parser(p, s) for p in paths if p and os.path.exists(p)]
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        dfs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_load)(s) for s in samples)
        return pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    @staticmethod
    def load_mutect_maf(maf_path: str, sample_id: str, source_label: str = "mutect2",
                    variant_group: str = "all"):
        """
        Load a Mutect2 MAF into a DataFrame with:
        sample, chrom, pos, ref, alt, key, vaf, source

        Only PASS (or '.') calls are kept if a FILTER column is present.

        Parameters
        ----------
        variant_group : "snv"   → SNP/DNP/TNP with strict single-base ref+alt
                "indel" → INS/DEL
                "all"   → no filtering (default)
        """
        df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)

        # --- PASS filter (MAF-level) ---
        if "FILTER" in df.columns:
            df = df[df["FILTER"].isin(["PASS", "."])].copy()

        # --- Variant type filter ---
        VT_MAP = {
            "snv":   ["SNP", "DNP", "TNP"],
            "indel": ["INS", "DEL"],
            "all":   None,
        }
        if variant_group not in VT_MAP:
            raise ValueError(f"variant_group must be one of {list(VT_MAP.keys())}, got '{variant_group}'")

        vt_filter = VT_MAP[variant_group]
        if vt_filter is not None:
            if "Variant_Type" not in df.columns:
                raise ValueError(f"{maf_path}: no Variant_Type column found")
            df = df[df["Variant_Type"].isin(vt_filter)].copy()

        if variant_group == "snv":
            df = df[
                (df["Reference_Allele"].str.len() == 1) &
                (df["Tumor_Seq_Allele2"].str.len() == 1)
            ].copy()

        # IMPORTANT: initialize with same index as (filtered) df
        out = pd.DataFrame(index=df.index)

        # Use your ONT sample ID for consistency with clairs_all / smrest_all
        out["sample"] = sample_id

        # Chrom/pos
        if "Chromosome" not in df.columns or "Start_Position" not in df.columns:
            raise ValueError(f"{maf_path}: missing Chromosome/Start_Position")

        out["chrom"] = df["Chromosome"].astype(str).map(SVmethods._normalize_chrom)
        out["pos"]   = df["Start_Position"].astype(int)

        # Ref / alt
        if "Reference_Allele" not in df.columns or "Tumor_Seq_Allele2" not in df.columns:
            raise ValueError(f"{maf_path}: missing Reference_Allele/Tumor_Seq_Allele2")

        out["ref"] = df["Reference_Allele"].astype(str)
        out["alt"] = df["Tumor_Seq_Allele2"].astype(str)

        # VAF
        vaf = None
        for col in ["t_vaf", "tumor_f", "Tumor_VAF"]:
            if col in df.columns:
                vaf = df[col].astype(float)
                break

        if vaf is None:
            if "t_alt_count" in df.columns and "t_ref_count" in df.columns:
                alt  = df["t_alt_count"].astype(float)
                refc = df["t_ref_count"].astype(float)
                denom = alt + refc
                with np.errstate(divide="ignore", invalid="ignore"):
                    vaf = alt / denom
            else:
                vaf = pd.Series(np.nan, index=df.index, dtype=float)

        out["vaf"] = vaf

        # Key aligned with your other callers
        out["ref"] = out["ref"].astype(str)
        out["alt"] = out["alt"].astype(str)
        out["key"] = list(zip(out["chrom"], out["pos"], out["ref"], out["alt"]))

        out["source"] = source_label

        return out
    
    @staticmethod
    def find_maf_for_sample(root_dir: str, sample_id: str, pattern: str = "*.maf") -> str:
        """
        Return a single MAF path under root_dir/sample_id.
        If multiple match, warn and take the first.
        """
        sample_dir = os.path.join(root_dir, sample_id)
        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(f"No directory for sample {sample_id}: {sample_dir}")

        mafs = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        if not mafs:
            raise FileNotFoundError(f"No MAFs matching {pattern} in {sample_dir}")

        if len(mafs) > 1:
            print(f"[WARN] Multiple MAFs for {sample_id}; using first:\n  {mafs[0]}")

        return mafs[0]
    
    @staticmethod
    def filter_snv_by_vaf(
        snv_df: pd.DataFrame,
        min_vaf: float | None = None,
        max_vaf: float | None = None,
        min_dp: int | None = None,
        min_alt: int | None = None,
        vaf_col: str = "vaf",
        dp_col: str = "dp",
        alt_col: str = "alt_depth",
    ) -> pd.DataFrame:
        """
        Return a filtered copy of an SNV DataFrame.

        Parameters
        ----------
        snv_df   : input SNV table
        min_vaf  : minimum VAF (default 0.05)
        max_vaf  : maximum VAF (default None)
        min_dp   : minimum total depth (default 10)
        min_alt  : minimum alt allele depth (default 3)
        vaf_col  : VAF column name (default 'vaf')
        dp_col   : depth column name (default 'dp')
        alt_col  : alt depth column name (default 'alt_depth')
        """
        if snv_df is None or snv_df.empty:
            return snv_df

        if vaf_col not in snv_df.columns:
            raise KeyError(f"VAF column '{vaf_col}' not found in DataFrame.")

        df = snv_df.copy()
        mask = df[vaf_col].notna()

        if min_vaf is not None:
            mask &= (df[vaf_col] >= float(min_vaf))
        if max_vaf is not None:
            mask &= (df[vaf_col] <= float(max_vaf))

        if min_dp is not None:
            if dp_col in df.columns:
                mask &= (df[dp_col].notna() & (df[dp_col] >= min_dp))
            else:
                print(f"[WARN] DP column '{dp_col}' not found, skipping min_dp filter.")

        if min_alt is not None:
            if alt_col in df.columns:
                mask &= (df[alt_col].notna() & (df[alt_col] >= min_alt))
            else:
                print(f"[WARN] Alt depth column '{alt_col}' not found, skipping min_alt filter.")

        return df.loc[mask].copy()
    
    @staticmethod
    def _norm_chrom(chrom, want_chr_prefix=None):
        """
        Normalize chrom labels so clairs_df and breakend indices match.
        If want_chr_prefix is True -> 'chr7', False -> '7', None -> infer.
        """
        c = str(chrom).strip()
        has_chr = c.lower().startswith("chr")
        base = c[3:] if has_chr else c

        # standardize MT/M
        if base.upper() in {"M", "MT"}:
            base = "M" if (want_chr_prefix is True) else "MT"

        if want_chr_prefix is None:
            return ("chr" + base) if has_chr else base
        return ("chr" + base) if want_chr_prefix else base

    @staticmethod
    def add_sv_distance_to_clairs_df(clairs_df: pd.DataFrame,
                                 be_index: dict,
                                 distance_col: str = "dist_to_sv",
                                 is_near_col: str | None = "near_sv_10kb",
                                 near_threshold: int = 10_000) -> pd.DataFrame:
        """
        For each variant in clairs_df (requires columns: sample, chrom, pos),
        compute distance to nearest SV breakend (union of callers) and add:
        - distance_col: distance in bp (NaN if no breakends on that chrom)
        - is_near_col: boolean, True if distance <= near_threshold (optional)
        """
        dists = []
        near_flags = []

        for row in clairs_df.itertuples(index=False):
            sample = row.sample
            chrom = row.chrom
            pos = int(row.pos)

            sample_be = be_index.get(sample)
            if sample_be is None:
                dists.append(np.nan)
                near_flags.append(False)
                continue

            breaks = sample_be.get(chrom)
            if breaks is None or breaks.size == 0:
                dists.append(np.nan)
                near_flags.append(False)
                continue

            d = SVmethods.nearest_distance_to_breakends(pos, breaks)
            dists.append(d)
            near_flags.append(bool(d <= near_threshold) if not np.isnan(d) else False)

        out = clairs_df.copy()
        out[distance_col] = dists
        if is_near_col is not None:
            out[is_near_col] = near_flags
        return out

    @staticmethod
    def add_sv_distance_to_clairs_df_percaller(
        clairs_df,
        be_index_by,                       # be_index_by[caller][svtype][sample][chrom] -> np.ndarray
        callers=None,
        svtypes=("BND","DEL","DUP","INV","INS","total"),
        dist_prefix="dist_to_sv",
        near_prefix="near_sv",
        near_threshold=10_000,
    ):
        """
        Adds columns like:
        dist_to_sv__{caller}__{svt}
        near_sv_10kb__{caller}__{svt}
        """
        if clairs_df.empty:
            return clairs_df

        callers_use = sorted(be_index_by.keys()) if callers is None else list(callers)

        out = clairs_df.copy()
        for caller in callers_use:
            if caller not in be_index_by:
                continue
            for svt in svtypes:
                if svt not in be_index_by[caller]:
                    continue

                dist_col = f"{dist_prefix}__{caller}__{svt}"
                near_col = f"{near_prefix}_{near_threshold//1000}kb__{caller}__{svt}"

                dists = []
                near_flags = []

                idx = be_index_by[caller][svt]  # [sample][chrom] -> breaks
                for row in out.itertuples(index=False):
                    sample = row.sample
                    chrom  = row.chrom
                    pos    = int(row.pos)

                    sample_be = idx.get(sample)
                    if sample_be is None:
                        dists.append(np.nan); near_flags.append(False); continue

                    breaks = sample_be.get(chrom)
                    if breaks is None or breaks.size == 0:
                        dists.append(np.nan); near_flags.append(False); continue

                    d = SVmethods.nearest_distance_to_breakends(pos, breaks)
                    dists.append(d)
                    near_flags.append(bool(d <= near_threshold) if not np.isnan(d) else False)

                out[dist_col] = dists
                out[near_col] = near_flags

        return out

    @staticmethod
    def summarize_sv_proximity_for_callerset(
        clairs_df,
        be_sorted,
        callers,
        svtypes=("BND", "DEL", "DUP", "INV", "INS"),
        near_threshold=10_000,
        sample_normalizer=None,
        label=None,
    ):
        """
        clairs_df: DataFrame of ClairS-TO variants with columns: sample, chrom, pos, overlap_category
        be_sorted: output of extract_breakends(...)
        callers: list of SV caller names to include in the union (e.g. ["nanomonsv","severus"])
        svtypes: SV types to include
        near_threshold: distance cutoff in bp (e.g. 10_000)
        sample_normalizer: function to map SV sample IDs -> variant sample IDs
        label: name to use for this callerset in the summary

        Returns: dict with summary metrics for this callerset.
        """

        if label is None:
            label = "+".join(callers)

        # Build union of breakends for this set of callers
        be_index = SVmethods.build_breakend_index_per_sample(
            be_sorted=be_sorted,
            callers=callers,
            svtypes=svtypes,
            sample_normalizer=sample_normalizer,
        )

        # Compute distance + near flag (but we only need the near flag here)
        dists = []
        near_flags = []
        cat_labels = []

        for row in clairs_df.itertuples(index=False):
            sample = row.sample
            chrom = row.chrom
            pos = int(row.pos)
            cat  = getattr(row, "overlap_category", None)

            sample_be = be_index.get(sample)
            if sample_be is None:
                dists.append(np.nan)
                near_flags.append(False)
                cat_labels.append(cat)
                continue

            breaks = sample_be.get(chrom)
            if breaks is None or breaks.size == 0:
                dists.append(np.nan)
                near_flags.append(False)
                cat_labels.append(cat)
                continue

            d = SVmethods.nearest_distance_to_breakends(pos, breaks)
            dists.append(d)
            near_flags.append(bool(d <= near_threshold) if not np.isnan(d) else False)
            cat_labels.append(cat)

        dists = np.array(dists)
        near_flags = np.array(near_flags)
        cat_labels = np.array(cat_labels, dtype=object)

        # Overall fraction near SV
        valid_mask = ~np.isnan(dists)
        if valid_mask.sum() == 0:
            overall_near = np.nan
        else:
            overall_near = near_flags[valid_mask].mean()

        # Split by overlap category, if present
        cats = ["Shared_with_Clair3", "ClairS_TO_only"]
        near_by_cat = {}
        for c in cats:
            m = valid_mask & (cat_labels == c)
            if m.sum() == 0:
                near_by_cat[c] = np.nan
            else:
                near_by_cat[c] = near_flags[m].mean()

        # Median distance (overall) as an extra summary metric
        if valid_mask.sum() == 0:
            med_dist = np.nan
        else:
            med_dist = np.nanmedian(dists[valid_mask])

        out = {
            "callers": "+".join(callers),
            "label": label,
            "near_threshold_bp": near_threshold,
            "near_frac_all": overall_near,
            "near_frac_shared": near_by_cat["Shared_with_Clair3"],
            "near_frac_clairs_to_only": near_by_cat["ClairS_TO_only"],
            "median_dist_all": med_dist,
            "n_variants": valid_mask.sum(),
        }
        return out

    @staticmethod
    def compute_sv_proximity_per_caller_summary(
        clairs_df,
        be_sorted,
        sv_callers,
        near_threshold=10_000,
        svtypes=("BND","DEL","DUP","INV","INS"),
        sample_normalizer=None,
    ):
        rows = []
        for caller in sv_callers:
            be_index_caller = SVmethods.build_breakend_index_per_sample(
                be_sorted=be_sorted,
                callers=[caller],
                svtypes=svtypes,
                sample_normalizer=sample_normalizer,
            )

            df_tmp = SNVmethods.add_sv_distance_to_clairs_df(
                clairs_df=clairs_df,
                be_index=be_index_caller,
                distance_col="dist_tmp",
                is_near_col="near_tmp",
                near_threshold=near_threshold
            )

            valid = df_tmp["dist_tmp"].notna()
            n_valid = int(valid.sum())

            # median distance over all valid
            if n_valid:
                median_dist = float(df_tmp.loc[valid, "dist_tmp"].median())
            else:
                median_dist = np.nan

            # --- ALL ClairS-TO variants (shared + ClairS_TO_only)
            m_all = valid
            n_all = int(m_all.sum())
            n_near_all = int(df_tmp.loc[m_all, "near_tmp"].sum())
            frac_all = (n_near_all / n_all) if n_all else np.nan

            # --- Shared subset (if overlap_category present)
            if "overlap_category" in df_tmp.columns:
                m_shared = valid & (df_tmp["overlap_category"] == "Shared_with_Clair3")
                n_shared = int(m_shared.sum())
                n_near_shared = int(df_tmp.loc[m_shared, "near_tmp"].sum())
                frac_shared = (n_near_shared / n_shared) if n_shared else np.nan
            else:
                n_shared = 0
                n_near_shared = 0
                frac_shared = np.nan

            # --- ClairS-TO-only subset
            if "overlap_category" in df_tmp.columns:
                m_only = valid & (df_tmp["overlap_category"] == "ClairS_TO_only")
                n_only = int(m_only.sum())
                n_near_only = int(df_tmp.loc[m_only, "near_tmp"].sum())
                frac_only = (n_near_only / n_only) if n_only else np.nan
            else:
                n_only = 0
                n_near_only = 0
                frac_only = np.nan

            rows.append({
                "caller": caller,
                "near_threshold_bp": near_threshold,

                # Distances
                "median_dist_all": median_dist,
                "n_valid": n_valid,

                # All ClairS-TO
                "near_frac_all": float(frac_all) if np.isfinite(frac_all) else np.nan,
                "n_all": n_all,
                "n_near_all": n_near_all,

                # Shared with Clair3
                "near_frac_shared": float(frac_shared) if np.isfinite(frac_shared) else np.nan,
                "n_shared": n_shared,
                "n_near_shared": n_near_shared,

                # ClairS-TO-only
                "near_frac_clairs_to_only": float(frac_only) if np.isfinite(frac_only) else np.nan,
                "n_clairs_to_only": n_only,
                "n_near_clairs_to_only": n_near_only,
            })

        return (pd.DataFrame(rows)
                .set_index("caller")
                .sort_values("near_frac_all", ascending=False))
    
    @staticmethod
    def plot_sv_proximity_dual_axis(summary_df,
                                frac_col="near_frac_all",
                                count_col="n_near_all",
                                title="SV proximity per caller (all ClairS-TO variants)",
                                frac_ylabel="Fraction ≤ 10 kb from a breakpoint",
                                count_ylabel="# variants ≤ 10 kb from a breakpoint",
                                rotate_xticks=45,
                                frac_pad=0.05):
        """
        Single-axis bar plot:
        y = count_col (e.g., n_near_all = # variants within 10 kb).
        """
        if summary_df is None or summary_df.empty:
            print("[INFO] Empty summary_df")
            return

        dfp = summary_df.copy().sort_values(frac_col, ascending=False)

        if count_col not in dfp.columns:
            raise KeyError(f"Missing count_col='{count_col}'. Available: {list(dfp.columns)}")

        callers = dfp.index.tolist()
        x = np.arange(len(callers))

        counts = dfp[count_col].astype(float).values

        fig, ax_left = plt.subplots(figsize=(8, 4.5))

        ax_left.bar(x, counts, width=0.8)
        ax_left.set_ylabel(count_ylabel)

        finite_c = np.isfinite(counts)
        if finite_c.any():
            ax_left.set_ylim(0, float(np.max(counts[finite_c])) * 1.05)

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(callers, rotation=rotate_xticks, ha="right")
        ax_left.set_title(title)
        ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sv_proximity_dual_axis_to_only(summary_df,
                                        frac_col,
                                        count_col="n_near_clairs_to_only",
                                        title="ClairS-TO-only SV callers SV breakend proximity",
                                        frac_ylabel="Frac. of variants ≤ 10 kb from a breakpoint",
                                        count_ylabel="# variants",
                                        rotate_xticks=45,
                                        frac_pad=0.05):
        """
        Single-axis bar plot for ClairS-TO-only:
        y = count_col (n_near_clairs_to_only = # ClairS-TO-only variants ≤ 10 kb).
        """
        if summary_df is None or summary_df.empty:
            print("[INFO] Empty summary_df")
            return

        dfp = summary_df.copy()

        if frac_col not in dfp.columns:
            raise KeyError(f"Missing frac_col='{frac_col}'. Available columns: {list(dfp.columns)}")

        if count_col not in dfp.columns:
            raise KeyError(f"Missing count_col='{count_col}'. Available columns: {list(dfp.columns)}")

        dfp = dfp.sort_values(frac_col, ascending=False)

        callers = dfp.index.tolist()
        x = np.arange(len(callers))

        fracs = dfp[frac_col].astype(float).values
        counts = dfp[count_col].astype(float).values

        print("counts (ClairS-TO-only within 10 kb):", counts)
        print("fractions (ClairS-TO-only):", fracs)

        fig, ax_left = plt.subplots(figsize=(7, 4))

        ax_left.set_ylabel(f"{count_ylabel}")
        finite_c = np.isfinite(counts)
        if finite_c.any():
            ax_left.set_ylim(0, float(np.max(counts[finite_c])) * 1.05)
        ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)

        ax_left.bar(x, counts, width=0.8)

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(callers, rotation=rotate_xticks, ha="right")
        ax_left.set_title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def count_Check(svcallersum):
        # 1) Confirm the columns you intend to use exist
        needed = [
            "near_frac_all", "n_near_all",
            "near_frac_clairs_to_only", "n_near_clairs_to_only",
        ]
        missing = [c for c in needed if c not in svcallersum.columns]
        print("Missing:", missing)

        # 2) Print the exact columns and a small view of the relevant ones
        cols = [c for c in svcallersum.columns if ("near" in c or "frac" in c or "valid" in c or "total" in c)]
        print("Columns:", cols)
        print(svcallersum[cols].head())

        # 3) Invariant checks (customize total/denominator col names if yours differ)
        # These should be TRUE if the summary is well-defined:
        #   n_near_clairs_to_only <= n_near_all
        #   near_frac_clairs_to_only <= 1
        #   near_frac_all <= 1
        bad = []

        if "n_near_all" in svcallersum.columns and "n_near_clairs_to_only" in sv_per_caller_summary.columns:
            bad1 = svcallersum["n_near_clairs_to_only"] > svcallersum["n_near_all"]
            if bad1.any():
                bad.append(("n_near_clairs_to_only > n_near_all", svcallersum[bad1][["n_near_all", "n_near_clairs_to_only"]]))

        for fc in ["near_frac_all", "near_frac_clairs_to_only"]:
            if fc in svcallersum.columns:
                badf = svcallersum[fc] > 1
                if badf.any():
                    bad.append((f"{fc} > 1", svcallersum[badf][[fc]]))

        print("Bad conditions:", [b[0] for b in bad])
        for name, df_bad in bad:
            print("\n", name)
            print(df_bad) 


    @staticmethod
    def compute_sv_proximity_per_caller_summary_smrest(
        smrest_df,
        be_sorted,
        sv_callers,
        near_threshold=10_000,
        svtypes=("BND","DEL","DUP","INV","INS"),
        sample_normalizer=None,
    ):
        """
        Compute, for each SV caller, how many smrest variants lie within `near_threshold`
        of that caller's breakends.

        Returns a DataFrame indexed by caller with columns:
        - near_threshold_bp
        - near_frac_all      (fraction of all smrest variants near SVs)
        - n_all              (total smrest variants considered)
        - n_near_all         (# smrest variants near SVs)
        """
        rows = []
        for caller in sv_callers:
            be_index_caller = SVmethods.build_breakend_index_per_sample(
                be_sorted=be_sorted,
                callers=[caller],
                svtypes=svtypes,
                sample_normalizer=sample_normalizer,
            )

            df_tmp = SNVmethods.add_sv_distance_to_clairs_df(
                clairs_df=smrest_df,
                be_index=be_index_caller,
                distance_col="dist_tmp",
                is_near_col="near_tmp",
                near_threshold=near_threshold
            )

            valid = df_tmp["dist_tmp"].notna()

            # ALL smrest variants
            m_all = valid
            n_all = int(m_all.sum())
            n_near_all = int(df_tmp.loc[m_all, "near_tmp"].sum())
            frac_all = (n_near_all / n_all) if n_all else np.nan

            rows.append({
                "caller": caller,
                "near_threshold_bp": near_threshold,

                "near_frac_all": float(frac_all) if np.isfinite(frac_all) else np.nan,
                "n_all": n_all,
                "n_near_all": n_near_all,
            })

        return (pd.DataFrame(rows)
                .set_index("caller")
                .sort_values("near_frac_all", ascending=False))
    
    @staticmethod
    def plot_two_set_venn(
        set_a,
        set_b,
        labels=("A", "B"),
        title=None,
        color_a="tab:blue",
        color_b="tab:orange",
        alpha=0.35,
    ):
        """
        Very simple 2-set Venn using overlapping circles.
        """

        A = set_a
        B = set_b

        only_a = len(A - B)
        only_b = len(B - A)
        ab     = len(A & B)

        fig, ax = plt.subplots(figsize=(5, 4))

        r = 1.0
        cA = mpatches.Circle((-0.5, 0), r, fill=True, linewidth=2,
                    edgecolor=color_a, facecolor=color_a, alpha=alpha)
        cB = mpatches.Circle(( 0.5, 0), r, fill=True, linewidth=2,
                    edgecolor=color_b, facecolor=color_b, alpha=alpha)

        ax.add_patch(cA)
        ax.add_patch(cB)

        ax.text(-1.0, 0.0, f"{only_a}", ha="center", va="center", fontsize=_fs(11))
        ax.text( 1.0, 0.0, f"{only_b}", ha="center", va="center", fontsize=_fs(11))
        ax.text( 0.0, 0.0, f"{ab}",     ha="center", va="center", fontsize=_fs(11))

        ax.text(-0.8, 1.2, labels[0], ha="center", va="center", fontsize=_fs(12))
        ax.text( 0.8, 1.2, labels[1], ha="center", va="center", fontsize=_fs(12))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("off")

        if title:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_three_set_venn(
        set_a,
        set_b,
        set_c,
        labels=("A", "B", "C"),
        title=None,
        color_a="tab:blue",   # ClairS-TO
        color_b="tab:orange", # smrest
        color_c="tab:red",    # Mutect2
        alpha=0.35,
    ):
        """
        Very simple 3-set Venn using three overlapping Circles and black text.

        set_a, set_b, set_c are Python sets of keys.
        labels: (label for A, label for B, label for C).
        """

        A = set_a
        B = set_b
        C = set_c

        only_a  = len(A - B - C)
        only_b  = len(B - A - C)
        only_c  = len(C - A - B)
        ab_only = len((A & B) - C)
        ac_only = len((A & C) - B)
        bc_only = len((B & C) - A)
        abc_all = len(A & B & C)

        fig, ax = plt.subplots(figsize=(6, 5))

        # Positions chosen so that intersection is roughly in the middle
        r = 1.0
        cA = mpatches.Circle((-0.6,  0.25), r, fill=True, linewidth=2,
                    edgecolor=color_a, facecolor=color_a, alpha=alpha)
        cB = mpatches.Circle(( 0.6,  0.25), r, fill=True, linewidth=2,
                    edgecolor=color_b, facecolor=color_b, alpha=alpha)
        cC = mpatches.Circle(( 0.0, -0.55), r, fill=True, linewidth=2,
                    edgecolor=color_c, facecolor=color_c, alpha=alpha)

        ax.add_patch(cA)
        ax.add_patch(cB)
        ax.add_patch(cC)

        # Region text (counts) – all black
        ax.text(-1.1,  0.25, f"{only_a}",  ha="center", va="center", fontsize=_fs(11), color="black")  # A only
        ax.text( 1.1,  0.25, f"{only_b}",  ha="center", va="center", fontsize=_fs(11), color="black")  # B only
        ax.text( 0.0, -1.1, f"{only_c}",  ha="center", va="center", fontsize=_fs(11), color="black")   # C only

        ax.text( 0.0,  0.55, f"{ab_only}", ha="center", va="center", fontsize=_fs(11), color="black")  # A∩B only
        ax.text(-0.7, -0.15, f"{ac_only}", ha="center", va="center", fontsize=_fs(11), color="black")  # A∩C only
        ax.text( 0.7, -0.15, f"{bc_only}", ha="center", va="center", fontsize=_fs(11), color="black")  # B∩C only

        ax.text( 0.0, -0.05, f"{abc_all}", ha="center", va="center", fontsize=_fs(11), color="black")  # A∩B∩C

        # Labels near each circle
        ax.text(-0.9,  1.2, labels[0], ha="center", va="center", fontsize=_fs(12), color="black")
        ax.text( 0.9,  1.2, labels[1], ha="center", va="center", fontsize=_fs(12), color="black")
        ax.text( 0.0, -1.6, labels[2], ha="center", va="center", fontsize=_fs(12), color="black")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("off")

        if title:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_venn_from_callsets(**callsets: pd.DataFrame):
        """
        Plot a Venn diagram from any number of callset DataFrames.
        
        Usage:
            SNVmethods.plot_venn_from_callsets(
                clairsto_all=clairsto_all_raw,
                clairs_all=clairs_all_raw,
                clair3_all=clair3_all_raw,
            )
        
        Parameter names are mapped to display labels automatically:
            clairsto_all -> "ClairS-TO"
            clairs_all   -> "ClairS"
            clair3_all   -> "Clair3"
            smrest_all   -> "smrest"
            mutect_all   -> "Mutect2"
        """
        NAME_MAP = {
            "clairsto_all": "ClairS-TO",
            "clairs_all":   "ClairS",
            "clair3_all":   "Clair3",
            "smrest_all":   "smrest",
            "mutect_all":   "Mutect2",
        }

        tool_sets = {}
        for param_name, df in callsets.items():
            if df is not None and not df.empty and "key" in df.columns:
                label = NAME_MAP.get(param_name, param_name)
                tool_sets[label] = set(df["key"])

        if len(tool_sets) < 2:
            print("[WARN] Fewer than two non-empty callsets; Venn diagram not meaningful.")
            return
        elif len(tool_sets) == 2:
            (name_a, set_a), (name_b, set_b) = tool_sets.items()
            print(f"[INFO] Drawing 2-set Venn: {name_a} vs {name_b}")
            SNVmethods.plot_two_set_venn(
                set_a, set_b,
                labels=(name_a, name_b),
                title=f"{name_a} vs {name_b} overlap",
                color_a="tab:blue",
                color_b="tab:orange",
            )
        else:
            names = list(tool_sets.keys())[:3]
            print(names)
            A, B, C = (tool_sets[n] for n in names)
            print(f"[INFO] Drawing 3-set Venn: {names[0]}, {names[1]}, {names[2]}")
            SNVmethods.plot_three_set_venn(
                A, B, C,
                labels=tuple(names),
                title="Variant overlap across callers",
                color_a="tab:blue",
                color_b="tab:orange",
                color_c="tab:red",
            )

    @staticmethod
    def build_proximity_locus_sets(
        dfs_by_caller: dict,
        window_bp: int = 20,
        chrom_col: str = "chrom",
        pos_col: str = "pos",
    ):
        """
        Group SNVs from multiple callers into 'proximity loci', such that:

        - Loci are built per chromosome.
        - Within a locus, the span (max_pos - min_pos) <= window_bp.
        - When adding a new event at position p:
            * If p - current_locus_min_pos <= window_bp, it joins the current locus.
            * Otherwise, a new locus is started.

        Parameters
        ----------
        dfs_by_caller : dict
            {caller_name: DataFrame} with at least [chrom_col, pos_col].
        window_bp : int
            Max allowed width of a locus in bp.
        chrom_col, pos_col : str
            Column names for chromosome and position.

        Returns
        -------
        locus_sets_by_caller : dict
            {caller_name: set of locus_ids in which this caller has ≥1 SNV}.
        locus_meta : list of dict
            Each dict: {"locus_id", "chrom", "start", "end", "callers"}.
        """
        # normalize caller names and keep only non-empty dataframes
        cleaned = {}
        for caller, df in dfs_by_caller.items():
            if df is None or df.empty:
                continue
            if chrom_col not in df.columns or pos_col not in df.columns:
                continue
            cleaned[caller] = df[[chrom_col, pos_col]].copy()

        locus_sets_by_caller = {caller: set() for caller in cleaned.keys()}
        locus_meta = []

        if not cleaned:
            return locus_sets_by_caller, locus_meta

        # All chromosomes present across callers (in normalized form)
        all_chroms = sorted(
            set(
                str(ch)
                for df in cleaned.values()
                for ch in df[chrom_col].unique()
            )
        )

        locus_id = 0

        for chrom in all_chroms:
            # Collect all positions from all callers for this chromosome
            merged_events = []
            for caller, df in cleaned.items():
                sub = df[df[chrom_col] == chrom]
                if sub.empty:
                    continue
                # deduplicate positions per caller to avoid noise
                for pos in np.sort(sub[pos_col].unique()):
                    merged_events.append((pos, caller))

            if not merged_events:
                continue

            # Sort by genomic position
            merged_events.sort(key=lambda x: x[0])

            # Build loci with max span <= window_bp
            current_start = None
            current_end = None
            current_callers = set()

            def _flush_current():
                nonlocal locus_id, locus_meta, locus_sets_by_caller
                if current_start is None:
                    return
                this_id = locus_id
                locus_id += 1
                locus_meta.append({
                    "locus_id": this_id,
                    "chrom": chrom,
                    "start": current_start,
                    "end": current_end,
                    "callers": sorted(current_callers),
                })
                for c in current_callers:
                    locus_sets_by_caller[c].add(this_id)

            for pos, caller in merged_events:
                if current_start is None:
                    # start first locus
                    current_start = pos
                    current_end = pos
                    current_callers = {caller}
                else:
                    # check span vs window_bp using start of locus
                    if pos - current_start <= window_bp:
                        # stays in same locus
                        current_end = pos
                        current_callers.add(caller)
                    else:
                        # close current locus, start a new one
                        _flush_current()
                        current_start = pos
                        current_end = pos
                        current_callers = {caller}

            # flush last locus on this chromosome
            _flush_current()

        return locus_sets_by_caller, locus_meta

    @staticmethod
    def collapse_snvs_by_proximity_window(df, window_bp=20):
        """
        Collapse SNVs within each tool into non-overlapping 'proximity loci' along each
        (sample, chrom) axis.

        Logic:
        - group by (sample, chrom)
        - sort by pos
        - start a locus at the first variant
        - keep adding variants to that locus as long as (current_pos - locus_start_pos) <= window_bp
        - once that condition is broken, close the locus and start a new one

        We represent each locus by *one* representative variant (the first in that locus)
        but keep all original columns for that representative row.

        Returns a new DataFrame (subset of rows from df), with no overlapping loci per
        (sample, chrom) and at most one row per 20 bp window.
        """
        if df is None or df.empty:
            return df.copy()

        required_cols = {"sample", "chrom", "pos"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"collapse_snvs_by_proximity_window: missing columns {missing}")

        kept_rows = []

        for (sample, chrom), sub in df.groupby(["sample", "chrom"], sort=False):
            sub = sub.sort_values("pos")
            current_start = None
            current_rows = []

            for _, row in sub.iterrows():
                pos = int(row["pos"])
                if current_start is None:
                    # start first locus
                    current_start = pos
                    current_rows = [row]
                else:
                    if pos - current_start <= window_bp:
                        # still within the same 20 bp locus
                        current_rows.append(row)
                    else:
                        # finish previous locus: keep one representative
                        kept_rows.append(current_rows[0])
                        # start new locus
                        current_start = pos
                        current_rows = [row]

            # flush last locus for this (sample, chrom)
            if current_rows:
                kept_rows.append(current_rows[0])

        # Build new DF from kept rows; preserve columns
        collapsed_df = pd.DataFrame(kept_rows).reset_index(drop=True)
        return collapsed_df

    @staticmethod
    def _norm_chr(x: str) -> str:
        s = str(x).strip()
        if s.lower().startswith("chr"):
            s = s[3:]
        return s

    @staticmethod
    def filter_snvs(df):
        """
        Keep only SNVs: len(ref) == 1 and len(alt) == 1.
        """
        if df is None or df.empty:
            return df
        df = df.copy()
        df["len_ref"] = df["ref"].astype(str).str.len()
        df["len_alt"] = df["alt"].astype(str).str.len()
        return df.loc[(df["len_ref"] == 1) & (df["len_alt"] == 1)].copy()

    @staticmethod
    def plot_snvs_around_egfr(
        clairs_all,
        smrest_all,
        mutect_all,
        genome_build="hg38",
        flank_bp=200_000,
    ):
        """
        Plot ALL SNVs around EGFR for:
        - ClairS-TO (blue)
        - smrest    (orange)
        - Mutect2   (red)

        No island information here: just SNV positions as vertical ticks.
        """

        # --- EGFR coordinates ---
        if genome_build == "hg38":
            egfr_chr   = "7"
            egfr_start = 55_018_820
            egfr_end   = 55_211_628
        elif genome_build == "hg19":
            egfr_chr   = "7"
            egfr_start = 55_086_710
            egfr_end   = 55_279_321
        else:
            raise ValueError("genome_build must be 'hg38' or 'hg19'")

        win_start = egfr_start - flank_bp
        win_end   = egfr_end   + flank_bp

        # --- SNV-only subsets ---
        clairs_snv = SNVmethods.filter_snvs(clairs_all)
        smrest_snv = SNVmethods.filter_snvs(smrest_all)
        mutect_snv = SNVmethods.filter_snvs(mutect_all)

        # --- filter SNVs to EGFR window ---
        def _filter_window(df):
            if df is None or df.empty:
                return df
            df = df.copy()
            mask_chr = df["chrom"].map(SNVmethods._norm_chr) == egfr_chr
            mask_pos = (df["pos"] >= win_start) & (df["pos"] <= win_end)
            return df.loc[mask_chr & mask_pos].copy()

        clairs_win = _filter_window(clairs_snv)
        smrest_win = _filter_window(smrest_snv)
        mutect_win = _filter_window(mutect_snv)

        print(f"EGFR window [{win_start:,} – {win_end:,}] on chr{egfr_chr}")
        print(f"  ClairS-TO SNVs in window: {0 if clairs_win is None else len(clairs_win)}")
        print(f"  smrest    SNVs in window: {0 if smrest_win is None else len(smrest_win)}")
        print(f"  Mutect2   SNVs in window: {0 if mutect_win is None else len(mutect_win)}")

        # --- build the plot ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        callers = [
            ("ClairS-TO", clairs_win, "tab:blue"),
            ("smrest",    smrest_win, "tab:orange"),
            ("Mutect2",   mutect_win, "tab:red"),
        ]

        for ax, (name, df_var, color) in zip(axes, callers):
            # SNVs: vertical tick marks
            if df_var is not None and not df_var.empty:
                xs = df_var["pos"].values
                ax.vlines(xs, 0, 1, color=color, linewidth=0.8)
                ax.set_ylim(0, 1.1)
            else:
                ax.set_ylim(0, 1.1)

            # anchor x-axis to EGFR window
            ax.set_xlim(win_start, win_end)   # <<< ADD THIS LINE

            # panel label
            ax.text(
                0.01, 0.85, name,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=_fs(11),
                color=color,
            )

            # EGFR gene body
            ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)

            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

        axes[-1].set_xlabel(f"Genomic position on chr{egfr_chr} (bp)")
        fig.suptitle(f"SNVs around EGFR ({genome_build})", y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

    @staticmethod
    def plot_snvs_around_gene(
        gene: "str | dict",
        clairs_all: pd.DataFrame,
        flank_bp: int = 200_000,
        save_dir: "str | None" = None,
        samples_per_fig: "int | None" = None,
    ):
        """
        Per-sample SNV tick plots around a gene locus.

        Always saves one PDF per sample ({gene}_SNV_{sample}.pdf).
        When samples_per_fig is set, also saves batch PDFs with all samples
        stacked as rows ({gene}_SNV_batch{n}.pdf).
        save_dir: base path — datetime suffix appended and folder created automatically.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if isinstance(gene, str):
            coords = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        gene_chr   = coords["chrom"]
        gene_start = coords["start"]
        gene_end   = coords["end"]
        win_start  = gene_start - flank_bp
        win_end    = gene_end   + flank_bp

        clairs_snv = SNVmethods.filter_snvs(clairs_all)
        if clairs_snv is None or clairs_snv.empty:
            print(f"  No ClairS-TO SNVs to plot for {gene_name}.")
            return

        mask_chr = clairs_snv["chrom"].map(SNVmethods._norm_chr) == gene_chr
        mask_pos = (clairs_snv["pos"] >= win_start) & (clairs_snv["pos"] <= win_end)
        df_win   = clairs_snv.loc[mask_chr & mask_pos]
        samples  = sorted(df_win["sample"].unique()) if not df_win.empty else []

        print(f"{gene_name} [{win_start:,}–{win_end:,}]  SNVs in window: {len(df_win)}")
        if not samples:
            print("  No variants to plot.")
            return

        for sample in samples:
            sub = df_win.loc[df_win["sample"] == sample]
            fig, ax = plt.subplots(figsize=(12, 1.8))
            if not sub.empty:
                ax.vlines(sub["pos"].values, 0, 1, color="tab:blue", linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_xlim(win_start, win_end)
            ax.set_yticks([])
            ax.set_xlabel(f"chr{gene_chr} position (bp)")
            ax.set_title(f"{gene_name}  —  {SNVmethods._sample_label(sample)}  (ClairS-TO SNVs)", fontsize=_fs(9))
            ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
            ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout()

            if actual_dir is not None:
                fpath = os.path.join(actual_dir, f"{gene_name}_SNV_{sample}.pdf")
                fig.savefig(fpath, bbox_inches="tight")
                print(f"  {fpath}")
                plt.close(fig)
            else:
                plt.show()

        # --- Batch figures (optional) ---
        if samples_per_fig is not None:
            batches   = [samples[i:i + samples_per_fig]
                         for i in range(0, len(samples), samples_per_fig)]
            n_batches = len(batches)
            for b_idx, batch in enumerate(batches):
                n_s  = len(batch)
                fig, axes = plt.subplots(n_s, 1, figsize=(12, max(2, 1.5 * n_s)), sharex=True)
                if n_s == 1:
                    axes = [axes]
                for ax, sample in zip(axes, batch):
                    sub = df_win.loc[df_win["sample"] == sample] if not df_win.empty else pd.DataFrame()
                    if not sub.empty:
                        ax.vlines(sub["pos"].values, 0, 1, color="tab:blue", linewidth=0.8)
                    ax.set_ylim(0, 1.1)
                    ax.set_xlim(win_start, win_end)
                    ax.set_ylabel(SNVmethods._sample_label(sample), fontsize=_fs(7))
                    ax.set_yticks([])
                    ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
                    ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
                    ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
                    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
                axes[-1].set_xlabel(f"chr{gene_chr} position (bp)")
                fig.suptitle(f"ClairS-TO SNVs — {gene_name}  "
                             f"(batch {b_idx + 1}/{n_batches})", y=1.01)
                plt.tight_layout()

                if actual_dir is not None:
                    fname = f"{gene_name}_SNV_batch{b_idx + 1:02d}_of_{n_batches:02d}.pdf"
                    fpath = os.path.join(actual_dir, fname)
                    fig.savefig(fpath, bbox_inches="tight")
                    print(f"  {fpath}")
                    plt.close(fig)
                else:
                    plt.show()

    @staticmethod
    def parse_coral_bed(bed_path):
        """
        Parse a CoRAL cycles.bed file.
        Returns a DataFrame with columns: chr, start, end, orientation, cycle_id, iscyclic, weight
        """
        df = pd.read_csv(
            bed_path,
            sep="\t",
            comment="#",
            names=["chr", "start", "end", "orientation", "cycle_id", "iscyclic", "weight"]
        )
        # Filter only cyclic segments
        df = df[df["iscyclic"] == True].copy()
        return df

    @staticmethod
    def get_all_coral_amplicons(coral_base_path, sample_id):
        """
        Find all amplicon bed files for a given sample.
        Returns a dict: {amplicon_number: DataFrame}
        """
        sample_dir = Path(coral_base_path) / sample_id
        if not sample_dir.exists():
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
        
        amplicons = {}
        for bed_file in sample_dir.glob(f"{sample_id}_merged_coordsorted_amplicon*_cycles.bed"):
            # Extract amplicon number from filename
            fname = bed_file.stem  # e.g., "sampleid_merged_coordsorted_amplicon1_cycles"
            amp_num = fname.split("_amplicon")[1].split("_cycles")[0]
            amplicons[int(amp_num)] = SNVmethods.parse_coral_bed(bed_file)
        
        return amplicons

    @staticmethod
    def linearize_ecdna_segments(segments_df):
        """
        Convert ecDNA segments to a linearized coordinate system.
        
        Returns:
            - linearized_df: segments with new 'linear_start' and 'linear_end' columns
            - total_length: total linearized length
        """
        segments = segments_df.copy()
        # Keep segments in their original order (already sorted by cycle in the bed file)
        segments = segments.reset_index(drop=True)
        
        segments["seg_length"] = segments["end"] - segments["start"]
        segments["linear_start"] = segments["seg_length"].cumsum().shift(1, fill_value=0)
        segments["linear_end"] = segments["linear_start"] + segments["seg_length"]
        
        return segments, segments["linear_end"].max()

    @staticmethod
    def map_snv_to_linear(snv_df, segments_df):
        """
        Map SNV positions to linearized ecDNA coordinates.
        
        Returns: DataFrame with SNVs that fall within ecDNA segments, 
                with added 'linear_pos' column
        """
        if snv_df is None or snv_df.empty:
            return pd.DataFrame()
        
        snv_df = snv_df.copy()
        snv_df["chrom_norm"] = snv_df["chrom"].map(SNVmethods._norm_chr)
        
        mapped_snvs = []
        
        for idx, seg in segments_df.iterrows():
            seg_chr = SNVmethods._norm_chr(seg["chr"])
            seg_start = seg["start"]
            seg_end = seg["end"]
            linear_start = seg["linear_start"]
            
            # Find SNVs in this segment
            mask = (
                (snv_df["chrom_norm"] == seg_chr) &
                (snv_df["pos"] >= seg_start) &
                (snv_df["pos"] <= seg_end)
            )
            
            seg_snvs = snv_df[mask].copy()
            if not seg_snvs.empty:
                # Map to linear coordinates
                seg_snvs["linear_pos"] = linear_start + (seg_snvs["pos"] - seg_start)
                seg_snvs["segment_idx"] = idx
                mapped_snvs.append(seg_snvs)
        
        if mapped_snvs:
            return pd.concat(mapped_snvs, ignore_index=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def plot_ecdna_snvs(
        clairs_all,
        coral_base_path,
        sample_id,
        amplicon_number=None,
    ):
        """
        Plot ClairS-TO SNVs along linearized ecDNA path.
        
        Args:
            clairs_all: DataFrame with all ClairS-TO variants
            coral_base_path: Path to CoRAL output directory
            sample_id: Sample ID to analyze
            amplicon_number: Specific amplicon to plot (if None, plot all found)
        """
        # Get ecDNA segments
        amplicons = SNVmethods.get_all_coral_amplicons(coral_base_path, sample_id)
        
        if not amplicons:
            print(f"No cyclic amplicons found for {sample_id}")
            return
        
        # If specific amplicon requested
        if amplicon_number is not None:
            if amplicon_number not in amplicons:
                print(f"Amplicon {amplicon_number} not found. Available: {list(amplicons.keys())}")
                return
            amplicons = {amplicon_number: amplicons[amplicon_number]}
        
        # Filter to SNVs only
        clairs_snv = SNVmethods.filter_snvs(clairs_all)
        
        # Plot each amplicon
        for amp_num, segments in amplicons.items():
            print(f"\n=== Amplicon {amp_num} ===")
            print(f"Number of segments: {len(segments)}")
            
            # Linearize segments
            segments_lin, total_length = SNVmethods.linearize_ecdna_segments(segments)
            
            # Map SNVs to linear coordinates
            mapped_snvs = SNVmethods.map_snv_to_linear(clairs_snv, segments_lin)
            
            print(f"Total ecDNA length: {total_length:,} bp")
            print(f"ClairS-TO SNVs on ecDNA: {len(mapped_snvs)}")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Plot SNVs as vertical lines
            if not mapped_snvs.empty:
                ax.vlines(
                    mapped_snvs["linear_pos"].values,
                    0, 1,
                    color="tab:blue",
                    linewidth=1.0,
                    alpha=0.7
                )
            
            # Mark segment boundaries and labels
            for idx, seg in segments_lin.iterrows():
                # Segment boundary
                ax.axvline(seg["linear_start"], color="red", linestyle="--", linewidth=1.5, alpha=0.6)
                
                # Segment label
                seg_mid = (seg["linear_start"] + seg["linear_end"]) / 2
                label = f"chr{SNVmethods._norm_chr(seg['chr'])}\n{seg['start']:,}-{seg['end']:,}"
                ax.text(
                    seg_mid, 1.05,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=_fs(8),
                    rotation=0,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7)
                )
                
                # Shade alternate segments
                if idx % 2 == 0:
                    ax.axvspan(seg["linear_start"], seg["linear_end"], 
                            color="gray", alpha=0.1)
            
            # Final boundary
            ax.axvline(total_length, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
            
            ax.set_xlim(0, total_length)
            ax.set_ylim(0, 1.3)
            ax.set_xlabel("Linearized ecDNA position (bp)", fontsize=_fs(11))
            ax.set_ylabel("SNVs", fontsize=_fs(11))
            ax.set_title(
                f"ClairS-TO SNVs along ecDNA path - {sample_id} - Amplicon {amp_num}",
                fontsize=_fs(12),
                fontweight="bold"
            )
            ax.grid(True, axis="x", linestyle=":", alpha=0.3)
            
            # Remove y-axis ticks (just visual indicator)
            ax.set_yticks([])
            
            plt.tight_layout()
            plt.show()

    @staticmethod
    def debug_snv_positions(clairs_all, sample_id, segments):
        """
        Check specific SNV positions around the segment ranges
        """
        print(f"\n  DEBUG: Detailed position check for {sample_id}")
        
        clairs_snv = SNVmethods.filter_snvs(clairs_all)
        sample_snvs = clairs_snv[clairs_snv['sample'] == sample_id].copy()
        sample_snvs["chrom_norm"] = sample_snvs["chrom"].map(SNVmethods._norm_chr)
        
        segments = segments.reset_index(drop=True)
        
        # Check first segment on chr7
        seg = segments.iloc[0]
        seg_chr = SNVmethods._norm_chr(seg["chr"])
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        print(f"\n  Segment 1: chr{seg_chr}:{seg_start:,}-{seg_end:,}")
        
        # Get all chr7 SNVs
        chr7_snvs = sample_snvs[sample_snvs["chrom_norm"] == "7"].copy()
        chr7_snvs = chr7_snvs.sort_values('pos')
        
        print(f"  Total chr7 SNVs: {len(chr7_snvs)}")
        
        # Find SNVs closest to the segment
        nearby_before = chr7_snvs[chr7_snvs['pos'] < seg_start].tail(10)
        nearby_after = chr7_snvs[chr7_snvs['pos'] > seg_end].head(10)
        in_range = chr7_snvs[(chr7_snvs['pos'] >= seg_start) & (chr7_snvs['pos'] <= seg_end)]
        
        print(f"\n  SNVs in segment range: {len(in_range)}")
        if len(in_range) > 0:
            print(in_range[['chrom', 'pos', 'ref', 'alt']])
        
        print(f"\n  Last 10 SNVs BEFORE segment (pos < {seg_start:,}):")
        if len(nearby_before) > 0:
            print(nearby_before[['chrom', 'pos', 'ref', 'alt']])
        
        print(f"\n  First 10 SNVs AFTER segment (pos > {seg_end:,}):")
        if len(nearby_after) > 0:
            print(nearby_after[['chrom', 'pos', 'ref', 'alt']])

    @staticmethod
    def plot_all_ecdna_snvs_grid(
        variants_all,
        coral_base_path,
        caller_name="ClairS-TO",  # NEW PARAMETER
        sample_ids=None,
        segments_per_row=4,
        sample_column='sample',
        skip_dirs={'sparsedata'},
    ):
        """
        Plot SNVs for ALL samples and ALL amplicons.
        
        Args:
            variants_all: DataFrame with all variants
            coral_base_path: Path to CoRAL output directory
            caller_name: Name of the caller to display in plot titles (e.g., 'ClairS-TO', 'Mutect2', 'smrest')
            sample_ids: List of sample IDs to process (if None, auto-detect from directory)
            segments_per_row: Number of segment boxes per row
            sample_column: Name of column in variants_all that contains sample IDs (or None to skip filtering)
            skip_dirs: Set of directory names to skip when auto-detecting
        """
        coral_path = Path(coral_base_path)
        
        # Auto-detect samples if not provided
        if sample_ids is None:
            all_dirs = [d.name for d in coral_path.iterdir() if d.is_dir()]
            sample_ids = [d for d in all_dirs if d not in skip_dirs]
            print(f"Auto-detected {len(sample_ids)} samples (skipped {skip_dirs}): {sample_ids}")
        
        # Filter to SNVs only
        variants_snv = SNVmethods.filter_snvs(variants_all)
        print(f"\nTotal SNVs after filtering: {len(variants_snv)}")
        
        # Check if sample column exists
        if sample_column and sample_column in variants_snv.columns:
            print(f"Sample column '{sample_column}' found. Unique values:")
            print(f"  {variants_snv[sample_column].unique()}")
            use_sample_filtering = True
        else:
            print(f"\n⚠ Sample column '{sample_column}' not found or set to None.")
            print(f"  Available columns: {list(variants_snv.columns)}")
            print(f"  Will use ALL SNVs for each sample (no filtering)")
            use_sample_filtering = False
        
        # Loop through each sample
        for sample_id in sample_ids:
            print(f"\n{'='*60}")
            print(f"Processing sample: {sample_id}")
            print(f"{'='*60}")
            
            try:
                # Get all amplicons for this sample
                amplicons = SNVmethods.get_all_coral_amplicons(coral_base_path, sample_id)
                
                if not amplicons:
                    print(f"  ⚠ No cyclic amplicons found for {sample_id}")
                    continue
                
                print(f"  Found {len(amplicons)} amplicon(s): {list(amplicons.keys())}")
                
                # Filter SNVs for this sample
                if use_sample_filtering:
                    sample_snvs = variants_snv[variants_snv[sample_column] == sample_id].copy()
                    print(f"  Total SNVs for {sample_id}: {len(sample_snvs)}")
                    
                    if len(sample_snvs) == 0:
                        print(f"  ⚠ WARNING: No SNVs found for sample '{sample_id}'")
                        print(f"    Skipping this sample...")
                        continue
                else:
                    sample_snvs = variants_snv.copy()
                    print(f"  Using all SNVs: {len(sample_snvs)}")
                
                # Plot each amplicon
                for amp_num, segments in sorted(amplicons.items()):
                    print(f"\n  --- Amplicon {amp_num} ---")
                    print(f"      Segments: {len(segments)}")
                    
                    segments = segments.reset_index(drop=True)
                    
                    # Calculate grid dimensions
                    n_segments = len(segments)
                    n_rows = int(np.ceil(n_segments / segments_per_row))
                    n_cols = min(segments_per_row, n_segments)
                    
                    # Create figure with subplots
                    fig, axes = plt.subplots(
                        n_rows, n_cols,
                        figsize=(n_cols * 3.5, n_rows * 1.2),
                        squeeze=False
                    )
                    
                    # Flatten axes for easy iteration
                    axes_flat = axes.flatten()
                    
                    total_snvs = 0
                    
                    for idx, seg in segments.iterrows():
                        ax = axes_flat[idx]
                        
                        seg_chr = SNVmethods._norm_chr(seg["chr"])
                        seg_start = seg["start"]
                        seg_end = seg["end"]
                        seg_length = seg_end - seg_start
                        
                        # Find SNVs in this segment
                        snv_df_norm = sample_snvs.copy()
                        snv_df_norm["chrom_norm"] = snv_df_norm["chrom"].map(SNVmethods._norm_chr)
                        
                        mask = (
                            (snv_df_norm["chrom_norm"] == seg_chr) &
                            (snv_df_norm["pos"] >= seg_start) &
                            (snv_df_norm["pos"] <= seg_end)
                        )
                        
                        seg_snvs = snv_df_norm[mask]
                        n_snvs = len(seg_snvs)
                        
                        # Debug: print if we find SNVs
                        if n_snvs > 0:
                            print(f"      Seg {idx+1} (chr{seg_chr}:{seg_start}-{seg_end}): {n_snvs} SNVs")
                        
                        total_snvs += n_snvs
                        
                        # Plot SNVs as vertical lines
                        if not seg_snvs.empty:
                            ax.vlines(
                                seg_snvs["pos"].values,
                                0, 1,
                                color="tab:blue",
                                linewidth=1.5,
                                alpha=0.8
                            )
                        
                        # Set limits and styling
                        ax.set_xlim(seg_start, seg_end)
                        ax.set_ylim(0, 1)
                        ax.set_yticks([])
                        
                        # Title with segment info
                        title = f"Seg {idx+1}: chr{seg_chr}\n{seg_start:,} - {seg_end:,}\n({seg_length:,} bp, {n_snvs} SNVs)"
                        ax.set_title(title, fontsize=_fs(8), pad=3)
                        
                        # Light grid
                        ax.grid(True, axis="x", linestyle=":", alpha=0.3)
                        
                        # Format x-axis
                        ax.ticklabel_format(style='plain', axis='x')
                        ax.tick_params(axis='x', labelsize=_fs(7))
                        
                        # Light background
                        ax.set_facecolor("#f9f9f9")
                    
                    # Hide unused subplots
                    for idx in range(n_segments, len(axes_flat)):
                        axes_flat[idx].axis('off')
                    
                    # Overall title - USING caller_name PARAMETER
                    fig.suptitle(
                        f"{caller_name} SNVs by ecDNA Segment\n{sample_id} - Amplicon {amp_num}\nTotal: {total_snvs} SNVs across {n_segments} segments",
                        fontsize=_fs(12),
                        fontweight="bold",
                        y=0.995
                    )
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    plt.show()
                    
                    print(f"      Total SNVs in amplicon: {total_snvs}")
            
            except Exception as e:
                print(f"  ✗ Error processing {sample_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    @staticmethod
    def call_mutation_islands(df_long,
                            caller_name,
                            max_gap=50,
                            min_events=2):
        """
        df_long: variants for ONE caller (already filtered to event_len >= 2).
                Must have columns: sample, chrom, pos.
        Returns: DataFrame of islands:
        sample, chrom, caller, start_pos, end_pos, n_events
        """
        islands = []

        # group by sample, chrom
        for (sample, chrom), sub in df_long.groupby(["sample", "chrom"]):
            sub = sub.sort_values("pos").reset_index(drop=True)
            if sub.empty:
                continue

            start_idx = 0

            for i in range(1, len(sub)):
                # gap to previous event
                gap = sub.loc[i, "pos"] - sub.loc[i - 1, "pos"]

                # if gap too big, close previous island
                if gap > max_gap:
                    size = i - start_idx
                    if size >= min_events:
                        islands.append({
                            "sample": sample,
                            "chrom": chrom,
                            "caller": caller_name,
                            "start_pos": int(sub.loc[start_idx, "pos"]),
                            "end_pos":   int(sub.loc[i - 1, "pos"]),
                            "n_events":  int(size),
                        })
                    start_idx = i

            # close the last run
            size = len(sub) - start_idx
            if size >= min_events:
                islands.append({
                    "sample": sample,
                    "chrom": chrom,
                    "caller": caller_name,
                    "start_pos": int(sub.loc[start_idx, "pos"]),
                    "end_pos":   int(sub.loc[len(sub) - 1, "pos"]),
                    "n_events":  int(size),
                })
        return pd.DataFrame(islands)
    
    @staticmethod
    def add_island_midpoint(df_islands):
        """
        Add 'mid_pos' = integer midpoint of [start_pos, end_pos] for each island.

        If df_islands is empty or does not have start_pos/end_pos (e.g., no islands),
        it is returned unchanged.
        """
        if df_islands is None or df_islands.empty:
            return df_islands.copy()

        if "start_pos" not in df_islands.columns or "end_pos" not in df_islands.columns:
            # Nothing to do; keep as-is
            return df_islands.copy()

        df = df_islands.copy()
        df["mid_pos"] = ((df["start_pos"] + df["end_pos"]) // 2).astype(int)
        return df       

    @staticmethod
    def annotate_island_support(ref_islands, other_islands, caller_label, max_dist=100):
        """
        For each island in ref_islands, add a boolean column:
        has_<caller_label>_support

        True if there's at least one island in other_islands on the same
        sample/chrom whose midpoint is within max_dist bp of ref midpoint.

        If other_islands is empty or missing required columns, all flags stay False.
        """
        ref = ref_islands.copy()
        col_name = f"has_{caller_label}_support"

        # Initialize column to False
        ref[col_name] = False

        # If other_islands is empty or None, just return with all False
        if other_islands is None or other_islands.empty:
            return ref

        # Require these columns in other_islands
        required_cols = {"sample", "chrom", "mid_pos"}
        missing = required_cols - set(other_islands.columns)
        if missing:
            print(
                f"[WARN] annotate_island_support: other_islands missing columns {missing} "
                f"for caller_label='{caller_label}'. Skipping support annotation for this caller."
            )
            return ref

        # Build per-sample/chrom lists of midpoints for other caller
        other_dict = {}
        for (sample, chrom), sub in other_islands.groupby(["sample", "chrom"]):
            mids = sub["mid_pos"].values
            if len(mids):
                other_dict[(sample, chrom)] = np.sort(mids)

        if not other_dict:
            # No usable islands for the other caller
            return ref

        # Now annotate each ref island
        for idx, row in ref.iterrows():
            key = (row["sample"], row["chrom"])
            mids = other_dict.get(key)
            if mids is None or mids.size == 0:
                continue

            mp = row["mid_pos"]
            j = np.searchsorted(mids, mp)

            dists = []
            if j > 0:
                dists.append(abs(mp - mids[j - 1]))
            if j < len(mids):
                dists.append(abs(mp - mids[j]))

            if dists and min(dists) <= max_dist:
                ref.at[idx, col_name] = True

        return ref

    @staticmethod
    def plot_long_events_around_egfr(
        clairs_long,
        smrest_long,
        mutect_long,
        genome_build="hg38",
        flank_bp=200_000,
        ):
        """
        For each caller, plot each >2 bp event as a vertical bar:
        x = variant position (POS)
        y = event length in bp (max(len(ref), len(alt)))

        Tracks:
        - ClairS-TO (blue)
        - smrest    (orange)
        - Mutect2   (red)

        Assumes clairs_long / smrest_long / mutect_long have:
        chrom, pos, ref, alt, event_len (or ref/alt to recompute).
        """

        # --- EGFR coordinates ---
        if genome_build == "hg38":
            egfr_chr   = "7"
            egfr_start = 55_018_820
            egfr_end   = 55_211_628
        elif genome_build == "hg19":
            egfr_chr   = "7"
            egfr_start = 55_086_710
            egfr_end   = 55_279_321
        else:
            raise ValueError("genome_build must be 'hg38' or 'hg19'")

        win_start = egfr_start - flank_bp
        win_end   = egfr_end   + flank_bp

        def _prepare_long(df):
            if df is None or df.empty:
                return df
            df = df.copy()
            # If you already have event_len, you can just use that
            if "event_len" not in df.columns:
                df["event_len"] = df.apply(
                    lambda r: max(len(str(r["ref"])), len(str(r["alt"]))),
                    axis=1
                )
            # keep only >2 bp
            df = df.loc[df["event_len"] >= 1].copy()
            # window filter
            mask_chr = df["chrom"].map(SNVmethods._norm_chr) == egfr_chr
            mask_pos = (df["pos"] >= win_start) & (df["pos"] <= win_end)
            return df.loc[mask_chr & mask_pos].copy()

        clairs_win  = _prepare_long(clairs_long)
        smrest_win  = _prepare_long(smrest_long)
        mutect_win  = _prepare_long(mutect_long)

        print(f"EGFR window [{win_start:,} – {win_end:,}] on chr{egfr_chr}")
        print(f"  ClairS-TO >2bp events in window: {0 if clairs_win is None else len(clairs_win)}")
        print(f"  smrest    >2bp events in window: {0 if smrest_win is None else len(smrest_win)}")
        print(f"  Mutect2   >2bp events in window: {0 if mutect_win is None else len(mutect_win)}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        callers = [
            ("ClairS-TO", clairs_win, "tab:blue"),
            ("smrest",    smrest_win, "tab:orange"),
            # ("Mutect2",   mutect_win, "tab:red"),
        ]

        for ax, (name, df_var, color) in zip(axes, callers):
            if df_var is not None and not df_var.empty:
                xs = df_var["pos"].values
                ys = df_var["event_len"].values

                ax.vlines(xs, 0, ys, color=color, linewidth=2)
                max_len = float(ys.max())
            else:
                max_len = 1.0

            ax.set_ylim(0, max_len * 1.1)
            ax.set_ylabel("Island length (bp)")

            # Fix x-range to EGFR window so EGFR stays centered
            ax.set_xlim(win_start, win_end)

            # panel label
            ax.text(
                0.01, 0.9, name,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=_fs(11),
                color=color,
            )

            # EGFR gene body
            ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)

            ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        axes[-1].set_xlabel(f"Genomic position on chr{egfr_chr} (bp)")
        fig.suptitle(f">1 bp variants around EGFR ({genome_build})", y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()



    @staticmethod
    def _norm_chr_sv(x: str) -> str:
        s = str(x).strip()
        if s.lower().startswith("chr"):
            s = s[3:]
        if s.upper() == "M":
            return "MT"
        return s

    @staticmethod
    def _collect_sv_vcfs_indexed(caller: str, sample_dir: str) -> list:
        """Return tabix-indexed .vcf.gz SV files (with .tbi present) for a sample."""
        search_dir = os.path.join(sample_dir, "somatic_SVs") if caller == "severus" else sample_dir
        if not os.path.isdir(search_dir):
            return []
        return sorted(
            p for p in glob.glob(os.path.join(search_dir, "*.vcf.gz"))
            if os.path.exists(p + ".tbi")
        )

    @staticmethod
    def _list_sample_dirs_sv(parent_dir):
        return sorted(
            d for d in glob.glob(os.path.join(parent_dir, "*"))
            if os.path.isdir(d)
        )

    @staticmethod
    def _collect_sv_vcfs_for_sample_sv(caller, sample_dir):
        """
        Match the logic in summarize_sv_counts:
        - Severus: somatic_SVs/*.vcf
        - Others : *.vcf in sample_dir
        """
        def _is_final_vcf(p):
            name = os.path.basename(p).lower()
            return "raw" not in name

        if caller == "severus":
            som_dir = os.path.join(sample_dir, "somatic_SVs")
            if not os.path.isdir(som_dir):
                print(f"[WARN] Expected somatic_SVs not found: {som_dir}")
                return []
            return sorted(filter(_is_final_vcf, glob.glob(os.path.join(som_dir, "*.vcf"))))
        else:
            return sorted(filter(_is_final_vcf, glob.glob(os.path.join(sample_dir, "*.vcf"))))

    @staticmethod
    def build_sv_all_for_egfr(
        root_folder: str,
        caller_folders: dict | None = None,
        genome_build: str = "hg38",
        flank_bp: int = 200_000,
    ):
        """
        Build a long table of SVs overlapping EGFR ± flank_bp, using the
        SAME callers and directory structure as summarize_sv_counts.

        Returns a DataFrame with columns:
        sample, caller, chrom, start, end, svtype
        """

        # EGFR coordinates
        if genome_build == "hg38":
            egfr_chr   = "7"
            egfr_start = 55_018_820
            egfr_end   = 55_211_628
        elif genome_build == "hg19":
            egfr_chr   = "7"
            egfr_start = 55_086_710
            egfr_end   = 55_279_321
        else:
            raise ValueError("genome_build must be 'hg38' or 'hg19'")

        win_start = egfr_start - flank_bp
        win_end   = egfr_end   + flank_bp

        if caller_folders is None:
            caller_folders = {
                "nanomonsv": "nanomonsv_out",
                "severus": "severus",
                "sniffles_v1": "sniffles1_dec10",
                "sniffles_v2": "sniffles2_matchingDecoil",
                "manta_v1.6": "manta",
                "manta": "manta_from_peter",
            }

        rows = []

        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(root_folder, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue

            for sample_dir in SNVmethods._list_sample_dirs_sv(caller_path):
                sample_name = os.path.basename(sample_dir.rstrip(os.sep))
                vcfs = SNVmethods._collect_sv_vcfs_for_sample_sv(caller, sample_dir)
                if not vcfs:
                    print(f"[INFO] No SV VCFs for {caller} / {sample_name}")
                    continue

                for vf in vcfs:
                    try:
                        reader = vcfpy.Reader.from_path(vf)
                    except Exception as e:
                        print(f"[WARN] Failed to open {vf}: {e}")
                        continue

                    for rec in reader:
                        # --- PASS filter ---
                        if not VariationAnalyzer.vcf_record_is_pass(rec):
                            continue
                        chrom = SNVmethods._norm_chr_sv(rec.CHROM)
                        if chrom != egfr_chr:
                            continue

                        svtype = rec.INFO.get("SVTYPE")
                        if isinstance(svtype, list):
                            svtype = svtype[0]
                        svtype = str(svtype).strip().upper() if svtype is not None else None
                        if svtype is None:
                            continue

                        start = int(rec.POS)

                        end_info = rec.INFO.get("END")
                        if end_info is not None:
                            try:
                                end = int(end_info)
                            except Exception:
                                end = start
                        else:
                            # INS/BND etc. often only have POS; treat as point event
                            end = start

                        # Overlap EGFR window?
                        if end < win_start or start > win_end:
                            continue

                        rows.append({
                            "sample": sample_name,
                            "caller": caller,
                            "chrom":  chrom,
                            "start":  start,
                            "end":    end,
                            "svtype": svtype,
                        })

        if not rows:
            print("[WARN] No SVs overlapping EGFR window.")
            return pd.DataFrame(columns=["sample","caller","chrom","start","end","svtype"])

        sv_all = pd.DataFrame(rows)
        return sv_all


    @staticmethod
    def build_sv_all_for_gene(
        gene: "str | dict",
        root_folder: str,
        caller_folders: "dict | None" = None,
        flank_bp: int = 200_000,
        min_sv_len: int = 10_000,
    ) -> pd.DataFrame:
        """
        Build a long table of SVs overlapping a gene locus ± flank_bp.
        BNDs always pass the size filter (no span). Other SVtypes require
        |end - start| >= min_sv_len (default 10 kb).

        gene: key from GENE_COORDS (e.g. "TBXT") or dict with chrom/start/end.
        Returns DataFrame with columns: sample, caller, chrom, start, end, svtype.
        """
        if isinstance(gene, str):
            coords = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        gene_chr   = coords["chrom"]
        gene_start = coords["start"]
        gene_end   = coords["end"]
        win_start  = gene_start - flank_bp
        win_end    = gene_end   + flank_bp

        if caller_folders is None:
            caller_folders = DEFAULT_CALLER_FOLDERS

        chr_chrom = f"chr{gene_chr}" if not gene_chr.startswith("chr") else gene_chr

        rows = []
        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(root_folder, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue

            for sample_dir in SNVmethods._list_sample_dirs_sv(caller_path):
                sample_name = os.path.basename(sample_dir.rstrip(os.sep))

                # Prefer tabix-indexed .vcf.gz; fall back to plain .vcf
                vcfs = SNVmethods._collect_sv_vcfs_indexed(caller, sample_dir)
                use_fetch = bool(vcfs)
                if not vcfs:
                    vcfs = SNVmethods._collect_sv_vcfs_for_sample_sv(caller, sample_dir)
                if not vcfs:
                    continue

                for vf in vcfs:
                    try:
                        reader = vcfpy.Reader.from_path(vf)
                    except Exception as e:
                        print(f"[WARN] Failed to open {vf}: {e}")
                        continue

                    record_iter = reader.fetch(chr_chrom, win_start, win_end) \
                                  if use_fetch else iter(reader)

                    try:
                        for rec in record_iter:
                            if not VariationAnalyzer.vcf_record_is_pass(rec):
                                continue
                            chrom = SNVmethods._norm_chr_sv(rec.CHROM)
                            if chrom != gene_chr:
                                continue

                            svtype = rec.INFO.get("SVTYPE")
                            if isinstance(svtype, list):
                                svtype = svtype[0]
                            svtype = str(svtype).strip().upper() if svtype is not None else None
                            if svtype is None:
                                continue

                            start = int(rec.POS)
                            end_info = rec.INFO.get("END")
                            if end_info is not None:
                                try:
                                    end = int(end_info)
                                except Exception:
                                    end = start
                            else:
                                # try SVLEN as fallback
                                svlen = rec.INFO.get("SVLEN")
                                if svlen is not None:
                                    try:
                                        svlen = svlen[0] if isinstance(svlen, list) else svlen
                                        end = start + abs(int(svlen))
                                    except Exception:
                                        end = start
                                else:
                                    end = start

                            if not use_fetch and (end < win_start or start > win_end):
                                continue

                            # size filter — BNDs always pass; zero-span (no END/SVLEN) excluded
                            if svtype != "BND" and abs(end - start) < min_sv_len:
                                continue
                            if svtype != "BND" and end == start:
                                continue

                            rows.append({
                                "sample": sample_name,
                                "caller": caller,
                                "chrom":  chrom,
                                "start":  start,
                                "end":    end,
                                "svtype": svtype,
                            })
                    except Exception as e:
                        print(f"[WARN] Skipping malformed record in {vf}: {e}")

        if not rows:
            print(f"[WARN] No SVs found overlapping {gene_name} window.")
            return pd.DataFrame(columns=["sample", "caller", "chrom", "start", "end", "svtype"])
        return pd.DataFrame(rows)


    _save_dir_cache: dict = {}   # base_path -> timestamped path, shared within a session

    # Matches ONTWGS12-35-240548-NP01 and extracts the patient number (35)
    _SAMPLE_LABEL_RE = re.compile(r'^ONTWGS\d+-(\d+)-\d+-NP\d+$')

    @staticmethod
    def _sample_label(sample: str) -> str:
        """
        Convert a full sample ID to a short plot label.
        ONTWGS12-35-240548-NP01  →  P-35
        Falls back to the original string if the pattern doesn't match.
        """
        m = SNVmethods._SAMPLE_LABEL_RE.match(sample)
        return f"P-{m.group(1)}" if m else sample

    @staticmethod
    def _gene_save_dir(save_dir: "str | None") -> "str | None":
        """
        Resolve a plot output folder, creating it and caching the result so every
        call in one session writes to the same place.

        With the module flag NEW_FOLDER_PER_RUN True (default) the base path gets
        a _YYYYmmdd_HHMMSS suffix, so each run is kept separately. Set it False —
        `variations_analyzer.NEW_FOLDER_PER_RUN = False` — to write straight into
        the base path and overwrite whatever is already there.
        """
        if save_dir is None:
            return None
        key = save_dir.rstrip("/")
        new_folder = _new_folder_per_run()
        cache_key = (key, new_folder)
        if cache_key in SNVmethods._save_dir_cache:
            return SNVmethods._save_dir_cache[cache_key]
        if new_folder:
            import datetime as _dt
            actual = key + "_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            note = ""
        else:
            actual = key
            note = "  (overwriting; VA_NEW_PLOT_FOLDER=0)"
        os.makedirs(actual, exist_ok=True)
        SNVmethods._save_dir_cache[cache_key] = actual
        print(f"Saving to: {actual}{note}")
        return actual

    @staticmethod
    def plot_gene_multi_track(
        gene: "str | dict",
        samples: list,
        clairs_all: pd.DataFrame,
        sv_all: pd.DataFrame,
        flank_bp: int = 200_000,
        sv_callers: tuple = ("sniffles_v1",),
        save_dir: "str | None" = None,
        samples_per_fig: "int | None" = None,
    ):
        """
        Per-sample multi-track plot: SNV tick row + one SV row per caller.

        Always saves one PDF per sample ({gene}_multitrack_{sample}.pdf).
        When samples_per_fig is set, also saves batch PDFs with that many samples
        as columns per figure ({gene}_multitrack_batch{n}.pdf).
        save_dir: base path — datetime suffix appended and folder created automatically.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if isinstance(gene, str):
            coords = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        gene_chr   = coords["chrom"]
        gene_start = coords["start"]
        gene_end   = coords["end"]
        win_start  = gene_start - flank_bp
        win_end    = gene_end   + flank_bp

        clairs_snv = SNVmethods.filter_snvs(clairs_all)

        def _snv(sample):
            df = clairs_snv
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.loc[df["sample"] == sample] if "sample" in df.columns else df
            return df.loc[
                (df["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                & (df["pos"] >= win_start) & (df["pos"] <= win_end)
            ]

        def _sv(sample, caller):
            if sv_all is None or sv_all.empty:
                return pd.DataFrame()
            df = sv_all.loc[(sv_all["sample"] == sample) & (sv_all["caller"] == caller)]
            return df.loc[
                (df["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                & (df["end"] >= win_start) & (df["start"] <= win_end)
            ]

        def _draw_sv_on_ax(ax, sample, caller):
            df_sv = _sv(sample, caller)
            ax.set_xlim(win_start, win_end)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(caller, fontsize=_fs(8))
            ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
            ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
            if not df_sv.empty:
                for svtype, color in SVTYPE_COLORS.items():
                    if svtype == "Total SV":
                        continue
                    for _, row in df_sv.loc[df_sv["svtype"] == svtype].iterrows():
                        s = max(row["start"], win_start)
                        e = min(row["end"], win_end) if not np.isnan(row["end"]) else row["start"]
                        if e <= s:
                            ax.hlines(0.5, max(s - 100, win_start), min(s + 100, win_end),
                                      color=color, linewidth=2)
                        else:
                            ax.hlines(0.5, s, e, color=color, linewidth=3)

        sv_legend_handles = [plt.plot([], [], color=c, linewidth=3)[0]
                             for sv, c in SVTYPE_COLORS.items() if sv != "Total SV"]
        sv_legend_labels  = [sv for sv in SVTYPE_COLORS if sv != "Total SV"]
        plt.close("all")

        n_rows = 1 + len(sv_callers)

        # --- Per-sample figures ---
        for sample in samples:
            fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.4 * n_rows), sharex=True)
            if n_rows == 1:
                axes = [axes]

            ax = axes[0]
            df_c = _snv(sample)
            if not df_c.empty:
                ax.vlines(df_c["pos"].values, 0, 1, color="tab:blue", linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_xlim(win_start, win_end)
            ax.set_ylabel("ClairS-TO", fontsize=_fs(8))
            ax.set_yticks([])
            ax.set_title(f"{gene_name}  —  {SNVmethods._sample_label(sample)}", fontsize=_fs(9))
            ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
            ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

            for i, caller in enumerate(sv_callers):
                _draw_sv_on_ax(axes[1 + i], sample, caller)

            axes[-1].set_xlabel(f"chr{gene_chr} position (bp)")
            axes[1].legend(sv_legend_handles, sv_legend_labels, title="SVTYPE",
                           frameon=False, loc="upper right", fontsize=_fs(7))
            plt.tight_layout()

            if actual_dir is not None:
                fpath = os.path.join(actual_dir, f"{gene_name}_multitrack_{sample}.pdf")
                fig.savefig(fpath, bbox_inches="tight")
                print(f"  {fpath}")
                plt.close(fig)
            else:
                plt.show()

        # --- Batch figures (optional) ---
        if samples_per_fig is not None:
            batches   = [samples[i:i + samples_per_fig]
                         for i in range(0, len(samples), samples_per_fig)]
            n_batches = len(batches)
            for b_idx, batch in enumerate(batches):
                n_s  = len(batch)
                fig, axes = plt.subplots(n_rows, n_s,
                                         figsize=(4 * n_s, 1.4 * n_rows),
                                         sharex=True)
                if n_s == 1:
                    axes = axes.reshape(n_rows, 1)
                if n_rows == 1:
                    axes = axes.reshape(1, n_s)

                for col, sample in enumerate(batch):
                    ax = axes[0, col]
                    df_c = _snv(sample)
                    if not df_c.empty:
                        ax.vlines(df_c["pos"].values, 0, 1, color="tab:blue", linewidth=0.8)
                    ax.set_ylim(0, 1.1)
                    ax.set_xlim(win_start, win_end)
                    ax.set_yticks([])
                    ax.set_title(SNVmethods._sample_label(sample), fontsize=_fs(8))
                    ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
                    ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
                    ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
                    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
                    if col == 0:
                        ax.set_ylabel("ClairS-TO", fontsize=_fs(8))
                    for i, caller in enumerate(sv_callers):
                        _draw_sv_on_ax(axes[1 + i, col], sample, caller)
                        if col == 0:
                            axes[1 + i, col].set_ylabel(caller, fontsize=_fs(8))
                    axes[-1, col].set_xlabel(f"chr{gene_chr} pos (bp)", fontsize=_fs(7))

                axes[1, 0].legend(sv_legend_handles, sv_legend_labels, title="SVTYPE",
                                  frameon=False, loc="upper right", fontsize=_fs(7))
                fig.suptitle(f"{gene_name}  —  batch {b_idx + 1}/{n_batches}", y=1.01)
                plt.tight_layout()

                if actual_dir is not None:
                    fname = f"{gene_name}_multitrack_batch{b_idx + 1:02d}_of_{n_batches:02d}.pdf"
                    fpath = os.path.join(actual_dir, fname)
                    fig.savefig(fpath, bbox_inches="tight")
                    print(f"  {fpath}")
                    plt.close(fig)
                else:
                    plt.show()

    @staticmethod
    def plot_gene_sv_snv_combined(
        gene: "str | dict",
        samples: list,
        clairs_all: pd.DataFrame,
        sv_all: pd.DataFrame,
        cnv_all: "pd.DataFrame | None" = None,
        flank_bp: int = 200_000,
        sv_callers: tuple = ("sniffles_v1",),
        save_dir: "str | None" = None,
        samples_per_fig: int = 4,
    ):
        """
        Batch figure with one row per patient. Each row = two panels sharing x-axis:
          - Top strip (narrow): SV spans coloured by SVTYPE
          - Bottom panel:       CNV copy-number profile with SNV ticks overlaid

        One PDF per batch saved to save_dir (datetime-stamped folder).
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if isinstance(gene, str):
            coords    = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords    = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        gene_chr   = coords["chrom"]
        gene_start = coords["start"]
        gene_end   = coords["end"]
        win_start  = gene_start - flank_bp
        win_end    = gene_end   + flank_bp

        clairs_snv = SNVmethods.filter_snvs(clairs_all)

        # ── per-sample data helpers ──────────────────────────────────────────
        def _snv(sample):
            if clairs_snv is None or clairs_snv.empty:
                return pd.DataFrame()
            df = clairs_snv.loc[clairs_snv["sample"] == sample] if "sample" in clairs_snv.columns else clairs_snv
            return df.loc[
                (df["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                & (df["pos"] >= win_start) & (df["pos"] <= win_end)
            ]

        def _sv(sample):
            if sv_all is None or sv_all.empty:
                return pd.DataFrame()
            df = sv_all.loc[sv_all["sample"] == sample]
            if "caller" in df.columns:
                df = df.loc[df["caller"].isin(sv_callers)]
            return df.loc[
                (df["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                & (df["end"] >= win_start) & (df["start"] <= win_end)
            ]

        def _cnv(sample):
            if cnv_all is None or cnv_all.empty:
                return pd.DataFrame()
            req = {"chrom", "start", "end", "cn", "sample"}
            if not req.issubset(cnv_all.columns):
                return pd.DataFrame()
            df = cnv_all.loc[cnv_all["sample"] == sample].copy()
            df["_chr"] = df["chrom"].astype(str).str.replace("chr", "", regex=False).str.strip()
            return df.loc[
                (df["_chr"] == gene_chr)
                & (df["end"] >= win_start) & (df["start"] <= win_end)
            ]

        # ── layout constants ─────────────────────────────────────────────────
        SV_H   = 0.35   # inches per patient for SV strip
        CNV_H  = 1.5    # inches per patient for CNV+SNV panel
        FIG_W  = 13
        CNV_GAIN_COL = "#ffbcaf"
        CNV_LOSS_COL = "#a5d5d8"
        CNV_NEUT_COL = "#cccccc"
        SNV_COL      = "#4da6ff"

        # ── batches ──────────────────────────────────────────────────────────
        batches = [samples[i:i + samples_per_fig]
                   for i in range(0, len(samples), samples_per_fig)]

        for batch_idx, batch in enumerate(batches):
            n = len(batch)
            # height_ratios: alternating [1, 4] per patient
            hr = []
            for _ in batch:
                hr += [1, 4]
            fig_h = n * (SV_H + CNV_H) + 0.6
            fig   = plt.figure(figsize=(FIG_W, fig_h))
            gs    = fig.add_gridspec(n * 2, 1,
                                     height_ratios=hr,
                                     hspace=0.08,
                                     left=0.07, right=0.78,
                                     top=0.97, bottom=0.06)

            axes_sv  = []
            axes_cnv = []
            for i in range(n):
                ax_sv  = fig.add_subplot(gs[i * 2])
                ax_cnv = fig.add_subplot(gs[i * 2 + 1], sharex=ax_sv)
                axes_sv.append(ax_sv)
                axes_cnv.append(ax_cnv)

            for i, sample in enumerate(batch):
                ax_sv  = axes_sv[i]
                ax_cnv = axes_cnv[i]
                label  = SNVmethods._sample_label(sample)
                is_last = (i == n - 1)

                # ── shared x setup ───────────────────────────────────────────
                for ax in (ax_sv, ax_cnv):
                    ax.set_xlim(win_start, win_end)
                    ax.axvspan(gene_start, gene_end, color="grey", alpha=0.10, zorder=0)
                    ax.axvline(gene_start, color="grey", linestyle="--", linewidth=0.6, zorder=1)
                    ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=0.6, zorder=1)
                    ax.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.4, zorder=0)
                    ax.xaxis.get_offset_text().set_visible(False)
                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
                    )

                # SV strip: no x labels; CNV panel: always show labels
                plt.setp(ax_sv.get_xticklabels(), visible=False)
                ax_sv.tick_params(axis="x", length=0)
                plt.setp(ax_cnv.get_xticklabels(), visible=True, fontsize=_fs(6), rotation=30, ha="right")

                if is_last:
                    ax_cnv.set_xlabel(f"chr{gene_chr} position (bp)", fontsize=_fs(7))

                # ── SV strip ─────────────────────────────────────────────────
                ax_sv.set_ylim(0, 1)
                ax_sv.set_yticks([])
                ax_sv.set_ylabel(label, fontsize=_fs(7), rotation=0,
                                 ha="right", va="center", labelpad=4)
                ax_sv.spines[["top", "right", "bottom", "left"]].set_visible(False)

                df_sv = _sv(sample)
                if not df_sv.empty:
                    for svtype, color in SVTYPE_COLORS.items():
                        if svtype == "Total SV":
                            continue
                        sub = df_sv.loc[df_sv["svtype"] == svtype]
                        for _, row in sub.iterrows():
                            s = max(row["start"], win_start)
                            e = row["end"] if not pd.isna(row["end"]) else row["start"]
                            e = min(e, win_end)
                            w = max(e - s, (win_end - win_start) * 0.003)
                            ax_sv.broken_barh([(s, w)], (0.15, 0.70),
                                              facecolors=color, edgecolors="none", alpha=0.85)

                # ── CNV + SNV panel ───────────────────────────────────────────
                df_cnv = _cnv(sample)
                cn_max = 5

                if not df_cnv.empty:
                    cn_vals = df_cnv["cn"].dropna()
                    cn_max  = max(6, int(cn_vals.max()) + 1)
                    for _, seg in df_cnv.iterrows():
                        s   = max(seg["start"], win_start)
                        e   = min(seg["end"],   win_end)
                        cn  = seg["cn"]
                        if cn > 2.5:
                            lc = CNV_GAIN_COL
                        elif cn < 1.5:
                            lc = CNV_LOSS_COL
                        else:
                            lc = CNV_NEUT_COL
                        ax_cnv.hlines(cn, s, e, colors=lc, linewidth=2.5, zorder=2)
                    ax_cnv.axhline(2, color="black", linewidth=0.5, linestyle=":", zorder=1)
                else:
                    ax_cnv.axhline(2, color="lightgrey", linewidth=0.8, linestyle="--", zorder=1)

                ax_cnv.set_ylim(0, cn_max)
                ax_cnv.set_yticks([0, 2, 4] if cn_max >= 4 else [0, 2])
                ax_cnv.tick_params(axis="y", labelsize=_fs(6), length=2)
                ax_cnv.set_ylabel("CN", fontsize=_fs(6), labelpad=2)
                ax_cnv.spines[["top", "right"]].set_visible(False)

                # SNV ticks — vertical lines drawn on the CN panel
                df_snv = _snv(sample)
                if not df_snv.empty:
                    ax_cnv.vlines(df_snv["pos"].values, 0, cn_max,
                                  color=SNV_COL, linewidth=0.8, alpha=0.6, zorder=4)

            # ── figure legend ────────────────────────────────────────────────
            sv_handles = [mpatches.Patch(color=c, label=sv)
                          for sv, c in SVTYPE_COLORS.items() if sv != "Total SV"]
            cnv_handles = [
                mlines.Line2D([], [], color=CNV_GAIN_COL, linewidth=2.5, label="CNV gain"),
                mlines.Line2D([], [], color=CNV_LOSS_COL, linewidth=2.5, label="CNV loss"),
                mlines.Line2D([], [], color=CNV_NEUT_COL, linewidth=2.5, label="CNV neutral"),
                mlines.Line2D([], [], color=SNV_COL, linewidth=1.0, label="SNV"),
            ]
            fig.legend(handles=sv_handles + cnv_handles,
                       fontsize=_fs(7), loc="upper left",
                       bbox_to_anchor=(0.80, 0.99), frameon=True, ncol=1)

            fig.suptitle(f"{gene_name}  —  batch {batch_idx + 1}", fontsize=_fs(9), y=0.995)

            if actual_dir is not None:
                fpath = os.path.join(actual_dir,
                                     f"{gene_name}_combined_batch{batch_idx + 1:02d}.pdf")
                fig.savefig(fpath, bbox_inches="tight")
                print(f"  {fpath}")
                plt.close(fig)
            else:
                plt.show()

    @staticmethod
    def plot_oncoprint(
        genes: list,
        samples: list,
        clairs_by_gene: dict,
        sv_by_gene: dict,
        cnv_all: "pd.DataFrame | None" = None,
        flank_bp: int = 200_000,
        snv_min_vaf: float = 0.05,
        save_dir: "str | None" = None,
    ):
        """
        Cohort-level oncoprint: columns = samples (sorted by alteration burden),
        rows = gene × alteration type (SNV / SV / CNV).

        clairs_by_gene: {'TBXT': clairs_tbxt_df, 'FN1': ..., 'SOX9': ...}
        sv_by_gene:     {'TBXT': sv_tbxt_df, ...}
        cnv_all:        full CNV DataFrame (Wakhan) or None — shown as 'pending' if absent
        save_dir:       base path — datetime appended, folder created automatically
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        ABSENT    = "#e8e8e8"
        PENDING   = "#d0d0d0"
        SNV_COL   = "#4da6ff"   # light cornflower blue — substitutions
        INDEL_COL = "#003f8a"   # deep navy — insertions / deletions
        CNV_GAIN  = "#ffbcaf"   # salmon — amplification / gain (cn > 2.5)
        CNV_LOSS  = "#a5d5d8"   # teal — deletion / loss (cn < 1.5)
        CNV_NEUT  = "#e8e8e8"   # same as ABSENT — diploid, not highlighted

        # ── Build alteration map ─────────────────────────────────────────────
        # alt_map[(gene, alt_type)][sample] = color | None
        alt_map = {}

        for gene in genes:
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp

            # SNV and INDEL (split by allele length)
            snv_row   = {}
            indel_row = {}
            df_c = clairs_by_gene.get(gene_name, pd.DataFrame())
            if not df_c.empty:
                df_c2 = df_c.copy()
                if snv_min_vaf > 0:
                    df_c2 = df_c2.loc[df_c2["vaf"].fillna(0) >= snv_min_vaf]
                pos_mask = (
                    (df_c2["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_c2["pos"] >= win_start) & (df_c2["pos"] <= win_end)
                )
                win_vars = df_c2.loc[pos_mask].copy()
                win_vars["_lref"] = win_vars["ref"].astype(str).str.len()
                win_vars["_lalt"] = win_vars["alt"].astype(str).str.len()
                is_snv   = (win_vars["_lref"] == 1) & (win_vars["_lalt"] == 1)
                for s in samples:
                    snv_row[s]   = SNV_COL   if not win_vars.loc[is_snv  & (win_vars["sample"] == s)].empty else None
                    indel_row[s] = INDEL_COL if not win_vars.loc[~is_snv & (win_vars["sample"] == s)].empty else None
            alt_map[(gene_name, "SNV")]   = {s: snv_row.get(s)   for s in samples}
            alt_map[(gene_name, "INDEL")] = {s: indel_row.get(s) for s in samples}

            # SV — color by dominant SVTYPE
            sv_row = {}
            df_sv = sv_by_gene.get(gene_name, pd.DataFrame())
            if not df_sv.empty:
                mask = (
                    (df_sv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_sv["end"] >= win_start) & (df_sv["start"] <= win_end)
                )
                sv_win = df_sv.loc[mask]
                for s in samples:
                    sub = sv_win.loc[sv_win["sample"] == s]
                    if not sub.empty:
                        dominant = sub["svtype"].value_counts().idxmax()
                        sv_row[s] = SVTYPE_COLORS.get(dominant, "#333333")
                    else:
                        sv_row[s] = None
            alt_map[(gene_name, "SV")] = {s: sv_row.get(s) for s in samples}

            # CNV — placeholder until Wakhan data arrives
            cnv_row = {}
            if cnv_all is not None and not cnv_all.empty:
                req = {"chrom", "start", "end", "cn", "sample"}
                if req.issubset(cnv_all.columns):
                    df_cnv = cnv_all.copy()
                    df_cnv["_chr"] = df_cnv["chrom"].astype(str).str.replace("chr", "", regex=False).str.strip()
                    cmask = (
                        (df_cnv["_chr"] == gene_chr)
                        & (df_cnv["end"]   >= win_start)
                        & (df_cnv["start"] <= win_end)
                    )
                    cnv_win = df_cnv.loc[cmask]
                    for s in samples:
                        sub = cnv_win.loc[cnv_win["sample"] == s]
                        if sub.empty:
                            cnv_row[s] = None
                        else:
                            mean_cn = sub["cn"].mean()
                            if mean_cn > 2.5:
                                cnv_row[s] = CNV_GAIN
                            elif mean_cn < 1.5:
                                cnv_row[s] = CNV_LOSS
                            else:
                                cnv_row[s] = None  # neutral — don't highlight
            alt_map[(gene_name, "CNV")] = {s: cnv_row.get(s) for s in samples}

        # ── Sort samples by total non-CNV alteration count ───────────────────
        def _burden(s):
            return sum(1 for (g, t), row in alt_map.items()
                       if row.get(s) is not None)
        sorted_samples = sorted(samples, key=_burden, reverse=True)

        # ── Row order ────────────────────────────────────────────────────────
        row_order = [(g if isinstance(g, str) else f"chr{GENE_COORDS[g]['chrom']}", alt)
                     for g in genes for alt in ("SNV", "INDEL", "SV", "CNV")]
        n_rows = len(row_order)
        n_cols = len(sorted_samples)

        # ── Figure layout ────────────────────────────────────────────────────
        cell_w = max(0.25, min(0.6, 14.0 / n_cols))
        fig_w  = max(10, n_cols * cell_w + 3.5)
        fig_h  = n_rows * 0.55 + 2.5

        fig = plt.figure(figsize=(fig_w, fig_h))
        # Top strip: alteration burden bar
        gs    = fig.add_gridspec(2, 1, height_ratios=[1, n_rows], hspace=0.05)
        ax_bar = fig.add_subplot(gs[0])
        ax     = fig.add_subplot(gs[1])

        # Burden bar
        burdens = [_burden(s) for s in sorted_samples]
        ax_bar.bar(range(n_cols), burdens, color="#555555", width=0.8)
        ax_bar.set_xlim(-0.5, n_cols - 0.5)
        ax_bar.set_xticks([])
        ax_bar.set_ylabel("# alt.\n(SNV+INDEL\n+SV+CNV)", fontsize=_fs(7))
        ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
        ax_bar.tick_params(axis="y", labelsize=_fs(6))

        # Oncoprint grid
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()

        for row_idx, (gene_name, alt_type) in enumerate(row_order):
            for col_idx, sample in enumerate(sorted_samples):
                color = alt_map.get((gene_name, alt_type), {}).get(sample)

                if color is not None:
                    _hatch = {"SNV": "....", "CNV_GAIN": "///", "CNV_LOSS": "\\\\\\\\"}.get(
                        alt_type if alt_type != "CNV" else (
                            "CNV_GAIN" if color == CNV_GAIN else "CNV_LOSS"
                        )
                    )
                    rect = mpatches.Rectangle(
                        (col_idx - 0.45, row_idx - 0.40), 0.9, 0.8,
                        linewidth=0, facecolor=color,
                        hatch=_hatch, edgecolor="white" if _hatch else None
                    )
                else:
                    rect = mpatches.Rectangle(
                        (col_idx - 0.45, row_idx - 0.40), 0.9, 0.8,
                        linewidth=0, facecolor=ABSENT
                    )
                ax.add_patch(rect)

        # Gene dividers
        for i, (gene_name, alt_type) in enumerate(row_order):
            if alt_type == "SNV" and i > 0:
                ax.axhline(i - 0.5, color="black", linewidth=1.2)

        # Axes labels
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{g}  {a}" for g, a in row_order], fontsize=_fs(8))
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([SNVmethods._sample_label(s) for s in sorted_samples], rotation=90, fontsize=_fs(6))
        ax.xaxis.tick_bottom()
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor=SNV_COL,   hatch="....", edgecolor="white", label="SNV"),
            mpatches.Patch(facecolor=INDEL_COL, label="INDEL (ins/del)"),
        ]
        for svtype, color in SVTYPE_COLORS.items():
            if svtype != "Total SV":
                legend_handles.append(mpatches.Patch(facecolor=color, label=f"SV: {svtype}"))
        legend_handles += [
            mpatches.Patch(facecolor=CNV_GAIN, hatch="///", edgecolor="white", label="CNV: gain (cn > 2.5)"),
            mpatches.Patch(facecolor=CNV_LOSS, hatch="\\\\\\\\", edgecolor="white", label="CNV: loss (cn < 1.5)"),
            mpatches.Patch(facecolor=ABSENT,   label="Not altered / neutral"),
        ]
        ax.legend(handles=legend_handles, fontsize=_fs(7), loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=True, borderaxespad=0)

        gene_labels = " / ".join(g if isinstance(g, str) else str(g) for g in genes)
        fig.suptitle(f"Cohort alteration summary — {gene_labels} loci  (n={n_cols})",
                     fontsize=_fs(11), y=1.01)

        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, "oncoprint_cohort.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()


    # ── Cohort statistics ────────────────────────────────────────────────────

    @staticmethod
    def _build_alt_matrix(genes, samples, clairs_by_gene, sv_by_gene,
                          cnv_all, flank_bp, snv_min_vaf):
        """
        Returns dict: (gene, alt_type) -> {sample: bool}
        alt_type in ("SNV", "INDEL", "SV", "CNV")
        """
        from scipy.stats import fisher_exact  # lazy import — only needed here
        alt = {}
        for gene in genes:
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp

            # SNV / INDEL
            df_c = clairs_by_gene.get(gene_name, pd.DataFrame())
            snv_set   = set()
            indel_set = set()
            if not df_c.empty:
                df_c2 = df_c.copy()
                if snv_min_vaf > 0:
                    df_c2 = df_c2.loc[df_c2["vaf"].fillna(0) >= snv_min_vaf]
                pmask = (
                    (df_c2["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_c2["pos"] >= win_start) & (df_c2["pos"] <= win_end)
                )
                win = df_c2.loc[pmask].copy()
                win["_lr"] = win["ref"].astype(str).str.len()
                win["_la"] = win["alt"].astype(str).str.len()
                is_snv = (win["_lr"] == 1) & (win["_la"] == 1)
                snv_set   = set(win.loc[is_snv,  "sample"].unique())
                indel_set = set(win.loc[~is_snv, "sample"].unique())

            alt[(gene_name, "SNV")]   = {s: s in snv_set   for s in samples}
            alt[(gene_name, "INDEL")] = {s: s in indel_set for s in samples}

            # SV
            sv_set = set()
            df_sv = sv_by_gene.get(gene_name, pd.DataFrame())
            if not df_sv.empty:
                smask = (
                    (df_sv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_sv["end"] >= win_start) & (df_sv["start"] <= win_end)
                )
                sv_set = set(df_sv.loc[smask, "sample"].unique())
            alt[(gene_name, "SV")] = {s: s in sv_set for s in samples}

            # CNV
            cnv_set = set()
            if cnv_all is not None and not cnv_all.empty:
                req = {"chrom", "start", "end", "cn", "sample"}
                if req.issubset(cnv_all.columns):
                    df_cnv = cnv_all.copy()
                    df_cnv["_chr"] = df_cnv["chrom"].astype(str).str.replace("chr", "", regex=False).str.strip()
                    cmask = (
                        (df_cnv["_chr"] == gene_chr)
                        & (df_cnv["end"] >= win_start) & (df_cnv["start"] <= win_end)
                        & ((df_cnv["cn"] > 2.5) | (df_cnv["cn"] < 1.5))
                    )
                    cnv_set = set(df_cnv.loc[cmask, "sample"].unique())
            alt[(gene_name, "CNV")] = {s: s in cnv_set for s in samples}

        return alt

    @staticmethod
    def plot_alteration_frequency(genes, samples, clairs_by_gene, sv_by_gene,
                                  cnv_all=None, flank_bp=200_000,
                                  snv_min_vaf=0.05, save_dir=None):
        """% of patients altered per gene × alteration type."""
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        alt = SNVmethods._build_alt_matrix(genes, samples, clairs_by_gene,
                                           sv_by_gene, cnv_all, flank_bp, snv_min_vaf)
        alt_types = ("SNV", "INDEL", "SV", "CNV")
        colors     = {"SNV": "#4da6ff", "INDEL": "#003f8a",
                      "SV": "#e07b39", "CNV": "#a0522d"}
        n          = len(samples)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        x      = np.arange(len(gene_names))
        width  = 0.18
        offsets = np.linspace(-(len(alt_types)-1)/2, (len(alt_types)-1)/2, len(alt_types)) * width

        fig, ax = plt.subplots(figsize=(max(6, len(gene_names) * 2.5), 4))
        for i, atype in enumerate(alt_types):
            freqs = [
                100 * sum(alt.get((g, atype), {}).get(s, False) for s in samples) / n
                for g in gene_names
            ]
            bars = ax.bar(x + offsets[i], freqs, width, label=atype,
                          color=colors[atype], edgecolor="white", linewidth=0.5)
            for bar, freq in zip(bars, freqs):
                if freq > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1, f"{freq:.0f}%",
                            ha="center", va="bottom", fontsize=_fs(6))

        ax.set_xticks(x)
        ax.set_xticklabels(gene_names, fontsize=_fs(10))
        ax.set_ylabel("% patients altered", fontsize=_fs(9))
        ax.set_ylim(0, 105)
        ax.set_title(f"Alteration frequency per gene  (n={n})", fontsize=_fs(10))
        ax.legend(fontsize=_fs(8), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_alteration_frequency.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_vaf_distribution(genes, samples, clairs_by_gene,
                              flank_bp=200_000, save_dir=None):
        """Violin plot of SNV VAFs per gene."""
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        data = {}
        for gene in genes:
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp
            df_c = clairs_by_gene.get(gene_name, pd.DataFrame())
            if df_c.empty or "vaf" not in df_c.columns:
                data[gene_name] = []
                continue
            snv = SNVmethods.filter_snvs(df_c)
            if snv is None or snv.empty:
                data[gene_name] = []
                continue
            pmask = (
                (snv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                & (snv["pos"] >= win_start) & (snv["pos"] <= win_end)
            )
            vafs = snv.loc[pmask, "vaf"].dropna().tolist()
            data[gene_name] = vafs

        fig, ax = plt.subplots(figsize=(max(5, len(gene_names) * 2), 4))
        positions = range(1, len(gene_names) + 1)
        vdata = [data[g] for g in gene_names]
        filled = [(v if v else [np.nan]) for v in vdata]

        parts = ax.violinplot(filled, positions=positions,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4da6ff")
            pc.set_alpha(0.6)

        for i, (gname, vals) in enumerate(zip(gene_names, vdata)):
            if vals:
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                           s=8, color="#003f8a", alpha=0.5, zorder=3)
            ax.text(i + 1, -0.06, f"n={len(vals)}", ha="center", fontsize=_fs(7),
                    transform=ax.get_xaxis_transform())

        ax.set_xticks(list(positions))
        ax.set_xticklabels(gene_names, fontsize=_fs(10))
        ax.set_ylabel("VAF", fontsize=_fs(9))
        ax.set_ylim(0, 1.05)
        ax.set_title("SNV VAF distribution per gene", fontsize=_fs(10))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_vaf_distribution.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_sv_size_distribution(genes, sv_by_gene, save_dir=None):
        """Histogram of SV sizes per type across all gene windows."""
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        frames = [sv_by_gene.get(g, pd.DataFrame()) for g in gene_names]
        df_all = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
                 if any(not f.empty for f in frames) else pd.DataFrame()

        if df_all.empty:
            print("[INFO] No SV data for size distribution.")
            return

        df_all = df_all.copy()
        df_all["size"] = (df_all["end"] - df_all["start"]).abs()
        # BNDs: keep those with a real span (cis-BND); point BNDs (size==0) kept separately
        df_non_bnd = df_all.loc[(df_all["svtype"] != "BND") & (df_all["size"] >= 10_000)]
        df_bnd     = df_all.loc[df_all["svtype"] == "BND"]
        df_all     = pd.concat([df_non_bnd, df_bnd], ignore_index=True)

        svtypes = [s for s in ["BND", "DEL", "DUP", "INS", "INV"] if s in df_all["svtype"].unique()]
        n_types = len(svtypes)
        fig, axes = plt.subplots(1, n_types, figsize=(max(8, n_types * 3), 4),
                                 sharey=False)
        if n_types == 1:
            axes = [axes]

        for ax, svtype in zip(axes, svtypes):
            raw = df_all.loc[df_all["svtype"] == svtype, "size"].values
            raw = raw[raw < 3_000_000_000]  # drop artifacts > 3 Gb
            point_bnd = int((raw == 0).sum())  # BNDs with no span info
            sizes = raw[raw > 0] / 1000        # bp → kb, skip zero-span

            if len(sizes) == 0:
                ax.set_title(f"{svtype}\n(n={point_bnd} point BNDs, no span)", fontsize=_fs(9))
                ax.spines[["top", "right"]].set_visible(False)
                continue

            log_sizes = np.log10(sizes)
            pad   = max(0.3, (log_sizes.max() - log_sizes.min()) * 0.1)
            lo    = log_sizes.min() - pad
            hi    = log_sizes.max() + pad
            bins  = np.linspace(lo, hi, 31)
            ax.hist(log_sizes, bins=bins, color=SVTYPE_COLORS.get(svtype, "#888"),
                    edgecolor="white", linewidth=0.4)
            ax.set_xlim(lo, hi)
            median_kb = np.median(sizes)
            ax.axvline(np.log10(median_kb), color="black", linestyle="--",
                       linewidth=0.8, label=f"median {median_kb:.0f} kb")
            tick_vals = [1, 5, 10, 50, 100, 500, 1_000, 5_000, 10_000,
                         50_000, 100_000, 500_000, 1_000_000]
            valid_ticks = [v for v in tick_vals if lo <= np.log10(v) <= hi]
            if not valid_ticks:
                valid_ticks = [int(round(median_kb))]
            ax.set_xticks([np.log10(v) for v in valid_ticks])
            ax.set_xticklabels([f"{v:,}" for v in valid_ticks], fontsize=_fs(7), rotation=30, ha="right")
            ax.set_xlabel("Size (kb, log scale)", fontsize=_fs(8))
            ax.set_ylabel("Count", fontsize=_fs(8))
            subtitle = f" (+{point_bnd} point BNDs)" if svtype == "BND" and point_bnd > 0 else ""
            ax.set_title(f"{svtype}{subtitle}", fontsize=_fs(9))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=_fs(7), frameon=False)

        fig.suptitle("SV size distribution across gene windows", fontsize=_fs(10))
        plt.tight_layout()

        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_sv_size_distribution.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_lollipop(genes, samples, clairs_by_gene,
                      flank_bp=200_000, save_dir=None):
        """SNV positions along each gene body, dot height = # patients sharing that site."""
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        fig, axes = plt.subplots(len(gene_names), 1,
                                 figsize=(13, 3 * len(gene_names)), squeeze=False)

        for ax, gene in zip(axes[:, 0], genes):
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp

            df_c = clairs_by_gene.get(gene_name, pd.DataFrame())
            if not df_c.empty:
                snv = SNVmethods.filter_snvs(df_c)
                if snv is not None and not snv.empty:
                    pmask = (
                        (snv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                        & (snv["pos"] >= win_start) & (snv["pos"] <= win_end)
                    )
                    pos_counts = snv.loc[pmask].groupby("pos")["sample"].nunique()
                    if not pos_counts.empty:
                        ax.vlines(pos_counts.index, 0, pos_counts.values,
                                  color="#4da6ff", linewidth=0.8, alpha=0.7)
                        ax.scatter(pos_counts.index, pos_counts.values,
                                   s=pos_counts.values * 12 + 10,
                                   color="#003f8a", zorder=3, alpha=0.8)
                        for pos, cnt in pos_counts[pos_counts >= 2].items():
                            ax.text(pos, cnt + 0.15, str(cnt),
                                    ha="center", va="bottom", fontsize=_fs(6))

            ax.axvspan(coords["start"], coords["end"],
                       color="grey", alpha=0.12, zorder=0)
            ax.axvline(coords["start"], color="grey", linestyle="--", linewidth=0.6)
            ax.axvline(coords["end"],   color="grey", linestyle="--", linewidth=0.6)
            ax.set_xlim(win_start, win_end)
            ax.set_ylim(0, ax.get_ylim()[1] + 0.5)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.set_ylabel("# patients", fontsize=_fs(8))
            ax.set_title(f"{gene_name}  —  SNV lollipop", fontsize=_fs(9))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=_fs(6), rotation=30)

        plt.tight_layout()
        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_lollipop.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_sv_breakpoints(genes, samples, sv_by_gene,
                            flank_bp=200_000, save_dir=None):
        """All SV spans stacked per gene — recurrent breakpoint regions stand out."""
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        fig, axes = plt.subplots(len(gene_names), 1,
                                 figsize=(13, 3 * len(gene_names)), squeeze=False)

        for ax, gene in zip(axes[:, 0], genes):
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp

            df_sv = sv_by_gene.get(gene_name, pd.DataFrame())
            if not df_sv.empty:
                smask = (
                    (df_sv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_sv["end"] >= win_start) & (df_sv["start"] <= win_end)
                )
                df_win = df_sv.loc[smask].copy()
                df_win = df_win.loc[(df_win["svtype"] == "BND") | ((df_win["end"] - df_win["start"]).abs() >= 10_000)]
                df_win = df_win.sort_values("start")
                for yi, (_, row) in enumerate(df_win.iterrows()):
                    s = max(row["start"], win_start)
                    e = min(row["end"],   win_end)
                    color = SVTYPE_COLORS.get(row["svtype"], "#888888")
                    ax.hlines(yi, s, e, colors=color, linewidth=1.5, alpha=0.7)

                ax.set_ylim(-1, len(df_win) + 1)
                ax.set_yticks([])
                ax.set_ylabel("SVs", fontsize=_fs(8))
            else:
                ax.set_yticks([])

            ax.axvspan(coords["start"], coords["end"],
                       color="grey", alpha=0.12, zorder=0)
            ax.axvline(coords["start"], color="grey", linestyle="--", linewidth=0.6)
            ax.axvline(coords["end"],   color="grey", linestyle="--", linewidth=0.6)
            ax.set_xlim(win_start, win_end)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.set_title(f"{gene_name}  —  SV breakpoints (all patients)", fontsize=_fs(9))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=_fs(6), rotation=30)

            sv_handles = [mpatches.Patch(color=c, label=sv)
                          for sv, c in SVTYPE_COLORS.items() if sv != "Total SV"]
            ax.legend(handles=sv_handles, fontsize=_fs(7), frameon=False,
                      loc="upper right")

        plt.tight_layout()
        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_sv_breakpoints.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_co_alteration_heatmap(genes, samples, clairs_by_gene, sv_by_gene,
                                   cnv_all=None, flank_bp=200_000,
                                   snv_min_vaf=0.05, save_dir=None):
        """
        Co-occurrence heatmap with Fisher's exact p-values.
        Upper triangle: odds ratio (log2). Lower triangle: -log10(p-value).
        """
        from scipy.stats import fisher_exact
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        alt = SNVmethods._build_alt_matrix(genes, samples, clairs_by_gene,
                                           sv_by_gene, cnv_all, flank_bp, snv_min_vaf)

        keys   = [k for k in alt if any(alt[k].values())]
        labels = [f"{g}\n{t}" for g, t in keys]
        n      = len(keys)
        n_s    = len(samples)

        or_mat  = np.zeros((n, n))
        p_mat   = np.ones((n, n))

        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                a = sum(alt[ki][s] and alt[kj][s] for s in samples)
                b = sum(alt[ki][s] and not alt[kj][s] for s in samples)
                c = sum(not alt[ki][s] and alt[kj][s] for s in samples)
                d = sum(not alt[ki][s] and not alt[kj][s] for s in samples)
                table = [[a, b], [c, d]]
                try:
                    odds, p = fisher_exact(table)
                except Exception:
                    odds, p = 1.0, 1.0
                if np.isinf(odds) or np.isnan(odds):
                    # degenerate table (zero cell) — clamp to max observable
                    log2_or = 5.0 if odds > 1 else -5.0
                else:
                    log2_or = np.log2(odds + 1e-6)
                    log2_or = np.clip(log2_or, -5.0, 5.0)
                or_mat[i, j] = log2_or
                p_mat[i, j]  = p

        fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.2) * 2, max(6, n * 1.2)))

        for ax, mat, title, cmap, fmt in zip(
            axes,
            [or_mat, -np.log10(p_mat + 1e-10)],
            ["log2(Odds Ratio)", "−log10(p-value)"],
            ["RdBu_r", "YlOrRd"],
            [".1f", ".1f"],
        ):
            vmax = np.nanmax(np.abs(mat[mat != 0])) if np.any(mat != 0) else 1
            im = ax.imshow(mat, cmap=cmap, vmin=-vmax if "RdBu" in cmap else 0,
                           vmax=vmax, aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=_fs(7), rotation=45, ha="right")
            ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=_fs(7))
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i,j]:{fmt}}", ha="center",
                            va="center", fontsize=_fs(5.5),
                            color="white" if abs(mat[i,j]) > vmax * 0.6 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=_fs(9))

        fig.suptitle("Co-alteration analysis", fontsize=_fs(11))
        plt.tight_layout()

        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_co_alteration_heatmap.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_burden_scatter(genes, samples, clairs_by_gene, sv_by_gene,
                            flank_bp=200_000, snv_min_vaf=0.05, save_dir=None):
        """
        Per-patient scatter: x = SNV count, y = SV count across all gene windows.
        Points coloured by which genes are altered.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        gene_names = [g if isinstance(g, str) else
                      f"chr{GENE_COORDS[g]['chrom']}:{GENE_COORDS[g]['start']}" for g in genes]

        snv_counts = {s: 0 for s in samples}
        sv_counts  = {s: 0 for s in samples}
        gene_hit   = {s: [] for s in samples}

        for gene in genes:
            coords    = GENE_COORDS[gene] if isinstance(gene, str) else gene
            gene_name = gene if isinstance(gene, str) else f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"
            gene_chr  = coords["chrom"]
            win_start = coords["start"] - flank_bp
            win_end   = coords["end"]   + flank_bp

            df_c = clairs_by_gene.get(gene_name, pd.DataFrame())
            if not df_c.empty:
                snv = SNVmethods.filter_snvs(df_c)
                if snv is not None and not snv.empty:
                    if snv_min_vaf > 0:
                        snv = snv.loc[snv["vaf"].fillna(0) >= snv_min_vaf]
                    pmask = (
                        (snv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                        & (snv["pos"] >= win_start) & (snv["pos"] <= win_end)
                    )
                    for s, cnt in snv.loc[pmask].groupby("sample").size().items():
                        if s in snv_counts:
                            snv_counts[s] += cnt

            df_sv = sv_by_gene.get(gene_name, pd.DataFrame())
            if not df_sv.empty:
                smask = (
                    (df_sv["chrom"].map(SNVmethods._norm_chr) == gene_chr)
                    & (df_sv["end"] >= win_start) & (df_sv["start"] <= win_end)
                )
                for s, cnt in df_sv.loc[smask].groupby("sample").size().items():
                    if s in sv_counts:
                        sv_counts[s] += cnt
                        if gene_name not in gene_hit[s]:
                            gene_hit[s].append(gene_name)

        palette = plt.cm.get_cmap("tab10", len(gene_names) + 1)
        gene_color = {g: palette(i) for i, g in enumerate(gene_names)}

        fig, ax = plt.subplots(figsize=(7, 6))

        for s in samples:
            x = snv_counts[s]
            y = sv_counts[s]
            hits = gene_hit[s]
            color = gene_color[hits[0]] if len(hits) == 1 else (
                "black" if len(hits) > 1 else "lightgrey"
            )
            ax.scatter(x, y, color=color, s=40, alpha=0.75, zorder=3,
                       edgecolors="white", linewidths=0.4)
            lbl = SNVmethods._sample_label(s)
            ax.annotate(lbl, (x, y), fontsize=_fs(5), alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        legend_handles = [mpatches.Patch(color=gene_color[g], label=g) for g in gene_names]
        legend_handles += [
            mpatches.Patch(color="black",     label="≥2 genes with SV"),
            mpatches.Patch(color="lightgrey", label="No SV in window"),
        ]
        ax.legend(handles=legend_handles, fontsize=_fs(7), frameon=True, loc="upper right")
        ax.set_xlabel("SNV count (all gene windows)", fontsize=_fs(9))
        ax.set_ylabel("SV count (all gene windows)", fontsize=_fs(9))
        ax.set_title(f"Patient alteration burden  (n={len(samples)})", fontsize=_fs(10))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        if actual_dir:
            fpath = os.path.join(actual_dir, "stats_burden_scatter.pdf")
            fig.savefig(fpath, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fpath}")
        else:
            plt.show()

    @staticmethod
    def plot_cohort_stats(genes, samples, clairs_by_gene, sv_by_gene,
                          cnv_all=None, flank_bp=200_000,
                          snv_min_vaf=0.05, save_dir=None):
        """Run all cohort statistics plots and save to save_dir."""
        print("── Alteration frequency ──────────────────")
        SNVmethods.plot_alteration_frequency(genes, samples, clairs_by_gene,
            sv_by_gene, cnv_all, flank_bp, snv_min_vaf, save_dir)
        print("── VAF distribution ──────────────────────")
        SNVmethods.plot_vaf_distribution(genes, samples, clairs_by_gene,
            flank_bp, save_dir)
        print("── SV size distribution ──────────────────")
        SNVmethods.plot_sv_size_distribution(genes, sv_by_gene, save_dir)
        print("── SNV lollipop ──────────────────────────")
        SNVmethods.plot_lollipop(genes, samples, clairs_by_gene,
            flank_bp, save_dir)
        print("── SV breakpoints ────────────────────────")
        SNVmethods.plot_sv_breakpoints(genes, samples, sv_by_gene,
            flank_bp, save_dir)
        print("── Co-alteration heatmap ─────────────────")
        SNVmethods.plot_co_alteration_heatmap(genes, samples, clairs_by_gene,
            sv_by_gene, cnv_all, flank_bp, snv_min_vaf, save_dir)
        print("── Burden scatter ────────────────────────")
        SNVmethods.plot_burden_scatter(genes, samples, clairs_by_gene,
            sv_by_gene, flank_bp, snv_min_vaf, save_dir)
        print("── Done ──────────────────────────────────")

    @staticmethod
    def plot_egfr_multi_track(
        samples,
        clairs_all,
        smrest_all,
        mutect_all,
        coverage_by_sample,   # can be dict or None
        sv_all,
        genome_build="hg38",
        flank_bp=200_000,
        sv_callers=("nanomonsv", "severus", "sniffles_v1", "sniffles_v2"),
        coverage_col="coverage",
    ):
        """
        For each sample (column), draw vertically:

        Row 0: ClairS-TO SNVs (ticks)
        Row 1: smrest SNVs (ticks)
        Row 2: Mutect2 SNVs (ticks)
        Row 3: coverage track (line)
        Row 4+: SV callers (one row per caller; horizontal spans by SVTYPE)
        Last row: EGFR bar

        All tracks are around EGFR ± flank_bp.
        """
        # Allow coverage_by_sample to be None (no coverage available yet)
        if coverage_by_sample is None:
            coverage_by_sample = {}

        # --- EGFR coordinates ---
        if genome_build == "hg38":
            egfr_chr   = "7"
            egfr_start = 55_018_820
            egfr_end   = 55_211_628
        elif genome_build == "hg19":
            egfr_chr   = "7"
            egfr_start = 55_086_710
            egfr_end   = 55_279_321
        else:
            raise ValueError("genome_build must be 'hg38' or 'hg19'")

        win_start = egfr_start - flank_bp
        win_end   = egfr_end   + flank_bp

        # Pre-filter SNVs to SNV-only
        clairs_snv  = SNVmethods.filter_snvs(clairs_all)
        smrest_snv  = SNVmethods.filter_snvs(smrest_all)
        mutect_snv  = SNVmethods.filter_snvs(mutect_all)

        def _filter_window(df, sample=None):
            if df is None or df.empty:
                return df
            df = df.copy()
            if sample is not None and "sample" in df.columns:
                df = df.loc[df["sample"] == sample].copy()
            mask_chr = df["chrom"].map(SNVmethods._norm_chr) == egfr_chr
            mask_pos = (df["pos"] >= win_start) & (df["pos"] <= win_end)
            return df.loc[mask_chr & mask_pos].copy()

        def _filter_cov(df_cov, sample):
            if df_cov is None or df_cov.empty:
                return df_cov
            df = df_cov.copy()
            mask_chr = df["chrom"].map(SNVmethods._norm_chr) == egfr_chr
            mask_pos = (df["pos"] >= win_start) & (df["pos"] <= win_end)
            return df.loc[mask_chr & mask_pos].copy()

        def _filter_sv(df_sv, sample, caller):
            if df_sv is None or df_sv.empty:
                return df_sv
            df = df_sv.copy()
            df = df.loc[(df["sample"] == sample) & (df["caller"] == caller)].copy()
            mask_chr = df["chrom"].map(SNVmethods._norm_chr) == egfr_chr
            # any overlap with window
            mask_win = (df["end"] >= win_start) & (df["start"] <= win_end)
            return df.loc[mask_chr & mask_win].copy()

        n_samples = len(samples)
        n_snv_rows = 3
        n_cov_rows = 0
        n_sv_rows  = len(sv_callers)
        n_bottom   = 0  # EGFR bar

        # n_rows = n_snv_rows + n_cov_rows + n_sv_rows + n_bottom
        n_rows = n_snv_rows + n_sv_rows   # 3 SNV rows + SV rows

        fig, axes = plt.subplots(
            n_rows, n_samples,
            figsize=(4 * n_samples, 1.4 * n_rows),
            sharex=True
        )

        # Ensure axes is 2D even if one sample
        if n_samples == 1:
            axes = axes.reshape(n_rows, 1)

        # Colors for SNV callers (consistent with earlier)
        SNV_COLORS = {
            "ClairS-TO": "tab:blue",
            "smrest":    "tab:orange",
            "Mutect2":   "tab:red",
        }

        # --- Main loop over samples / rows ---
        for col_idx, sample in enumerate(samples):
            # Row indices
            row_clairs  = 0
            row_smrest  = 1
            row_mutect  = 2
            # row_cov     = 3
            first_sv_row = 3
            # bottom_row   = n_rows - 1

            # 1) ClairS-TO SNVs
            ax = axes[row_clairs, col_idx]
            df_c = _filter_window(clairs_snv, sample=sample)
            if df_c is not None and not df_c.empty:
                xs = df_c["pos"].values
                ax.vlines(xs, 0, 1, color=SNV_COLORS["ClairS-TO"], linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_xlim(win_start, win_end)
            ax.set_ylabel("ClairS-TO") 
            ax.set_title(sample)  
            # ax.text(
            #     0.01, 0.85, f"{sample}\nClairS-TO",
            #     transform=ax.transAxes,
            #     ha="left", va="center", fontsize=_fs(9), color=SNV_COLORS["ClairS-TO"]
            # )
            ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)
            ax.set_yticks([])
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

            # 2) smrest SNVs
            ax = axes[row_smrest, col_idx]
            df_s = _filter_window(smrest_snv, sample=sample)
            if df_s is not None and not df_s.empty:
                xs = df_s["pos"].values
                ax.vlines(xs, 0, 1, color=SNV_COLORS["smrest"], linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_xlim(win_start, win_end)
            ax.set_ylabel("smrest")
            # ax.text(
            #     0.01, 0.85, "smrest",
            #     transform=ax.transAxes,
            #     ha="left", va="center", fontsize=_fs(9), color=SNV_COLORS["smrest"]
            # )
            ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)
            ax.set_yticks([])
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

            # 3) Mutect2 SNVs
            ax = axes[row_mutect, col_idx]
            df_m = _filter_window(mutect_snv, sample=sample)
            if df_m is not None and not df_m.empty:
                xs = df_m["pos"].values
                ax.vlines(xs, 0, 1, color=SNV_COLORS["Mutect2"], linewidth=0.8)
            ax.set_ylim(0, 1.1)
            ax.set_xlim(win_start, win_end)
            ax.set_ylabel("Mutect2")
            # ax.text(
            #     0.01, 0.85, "Mutect2",
            #     transform=ax.transAxes,
            #     ha="left", va="center", fontsize=_fs(9), color=SNV_COLORS["Mutect2"]
            # )
            ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)
            ax.set_yticks([])
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

            # 4) Coverage
            # ax = axes[row_cov, col_idx]
            # cov_df = coverage_by_sample.get(sample)
            # cov_win = _filter_cov(cov_df, sample) if cov_df is not None else None
            # if cov_win is not None and not cov_win.empty:
            #     ax.plot(cov_win["pos"].values,
            #             cov_win[coverage_col].values,
            #             linewidth=1)
            # ax.set_xlim(win_start, win_end)
            # ax.set_ylabel("Coverage")
            # ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
            # ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
            # ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)
            # ax.grid(True, axis="y", linestyle=":", alpha=0.4)

            # 5) SV callers (rows 4..4+len(sv_callers)-1)
            for i, caller in enumerate(sv_callers):
                row_idx = first_sv_row + i
                ax = axes[row_idx, col_idx]
                df_sv = _filter_sv(sv_all, sample, caller)

                ax.set_xlim(win_start, win_end)
                ax.set_ylim(0, 1)
                ax.set_yticks([])

                if df_sv is not None and not df_sv.empty:
                    # For each SVTYPE, draw spans or ticks
                    for svtype, color in SVTYPE_COLORS.items():
                        if svtype == "Total SV":
                            continue
                        sub = df_sv.loc[df_sv["svtype"] == svtype]
                        if sub.empty:
                            continue

                        for _, row in sub.iterrows():
                            start = max(row["start"], win_start)
                            end   = min(row["end"],   win_end) if not np.isnan(row["end"]) else row["start"]

                            # If no span (e.g., INS/BND), draw a short segment or tick
                            if end <= start:
                                # short 100 bp stub or 0-length
                                stub = 100
                                ax.hlines(0.5,
                                        max(start - stub, win_start),
                                        min(start + stub, win_end),
                                        color=color,
                                        linewidth=2)
                            else:
                                ax.hlines(0.5, start, end, color=color, linewidth=3)

                # EGFR shading
                ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.2)
                ax.axvline(egfr_start, color="grey", linestyle="--", linewidth=1)
                ax.axvline(egfr_end,   color="grey", linestyle="--", linewidth=1)

                if col_idx == 0:
                    ax.set_ylabel(caller)
                if i == len(sv_callers) - 1:
                    ax.set_xlabel(f"chr{egfr_chr} position (bp)")

            # 6) Bottom EGFR bar
            # ax = axes[bottom_row, col_idx]
            # ax.set_xlim(win_start, win_end)
            # ax.set_ylim(0, 1)
            # ax.hlines(0.5, egfr_start, egfr_end, color="black", linewidth=4)
            # ax.text(
            #     0.5 * (egfr_start + egfr_end),
            #     0.7,
            #     "EGFR",
            #     ha="center",
            #     va="bottom",
            #     fontsize=_fs(10),
            # )
            # # keep grey region too, so you can decide later
            # ax.axvspan(egfr_start, egfr_end, color="grey", alpha=0.15)
            # ax.set_yticks([])
            # ax.grid(False)

            # # X-label only on bottom row
            # ax.set_xlabel(f"chr{egfr_chr} position (bp)")
        

        # Optionally add a legend for SVTYPE colors on the first SV row, first sample
        first_sv_ax = axes[first_sv_row, 0]
        handles = []
        labels = []
        for svtype, color in SVTYPE_COLORS.items():
            if svtype == "Total SV":
                continue
            h = first_sv_ax.plot([], [], color=color, linewidth=3)[0]
            handles.append(h)
            labels.append(svtype)
        first_sv_ax.legend(
            handles,
            labels,
            title="SVTYPE",
            frameon=False,
            loc="upper right",
            fontsize=_fs(8),
        )

        fig.suptitle(f"EGFR locus: SNVs and SVs", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @staticmethod
    def build_chr_sample_count_matrix(df, chrom_order=None):
        """
        Returns a DataFrame with:
        rows   = chromosomes
        cols   = samples
        values = SNV counts
        """
        mat = (
            df
            .groupby(["chrom", "sample"])
            .size()
            .unstack(fill_value=0)
        )

        if chrom_order is not None:
            mat = mat.reindex(chrom_order)

        return mat


#############################################
#### METHODS RELATED TO CNV CALLERS ONLY ####
#############################################
class CNVmethods:
    """
    Copy-number variation analysis using Coral / CNVkit output or Wakhan output.

    Expected directory layout (one subfolder per sample):
        {root_folder}/{sample}/{sample}.minimap2.call.cns   ← Coral/CNVkit
        {root_folder}/{sample}/                              ← Wakhan (TBD)
    """

    @staticmethod
    def load_coral_cnv(root_folder: str, sample: str) -> pd.DataFrame:
        """
        Load CNVkit `.call.cns` segments for one sample from Coral output.

        Returns DataFrame with columns: chrom, start, end, log2, cn, sample
        (plus any extra CNVkit columns).
        """
        sample_dir = os.path.join(root_folder, sample)
        if not os.path.isdir(sample_dir):
            print(f"[WARN] Coral output dir not found: {sample_dir}")
            return pd.DataFrame()

        candidates = glob.glob(os.path.join(sample_dir, "*.call.cns"))
        if not candidates:
            print(f"[WARN] No .call.cns file found in {sample_dir}")
            return pd.DataFrame()

        dfs = []
        for path in candidates:
            try:
                df = pd.read_csv(path, sep="\t", comment="#", header=0)
                df.columns = [c.strip().lower() for c in df.columns]
                # Normalise column names: CNVkit uses 'chromosome' or 'chrom'
                if "chromosome" in df.columns and "chrom" not in df.columns:
                    df = df.rename(columns={"chromosome": "chrom"})
                # Ensure numeric cn
                if "cn" in df.columns:
                    df["cn"] = pd.to_numeric(df["cn"], errors="coerce")
                df["sample"] = sample
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Could not parse {path}: {e}")

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def load_coral_cnv_all(root_folder: str, samples: list) -> pd.DataFrame:
        """Load Coral CNV for all samples and concatenate."""
        frames = [CNVmethods.load_coral_cnv(root_folder, s) for s in samples]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print("[WARN] No Coral CNV data loaded for any sample.")
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def load_wakhan_cnv(root_folder: str, sample: str) -> pd.DataFrame:
        """
        Load Wakhan CNV segments for one sample from {root_folder}/{sample}/.

        NOTE: column names are placeholders — update once Wakhan output is available.
        """
        sample_dir = os.path.join(root_folder, sample)
        if not os.path.isdir(sample_dir):
            print(f"[WARN] Wakhan output dir not found: {sample_dir}")
            return pd.DataFrame()

        candidates = (
            glob.glob(os.path.join(sample_dir, "*.bed"))
            + glob.glob(os.path.join(sample_dir, "*.tsv"))
            + glob.glob(os.path.join(sample_dir, "*.txt"))
        )
        if not candidates:
            print(f"[WARN] No Wakhan CNV files found in {sample_dir}")
            return pd.DataFrame()

        dfs = []
        for path in candidates:
            try:
                df = pd.read_csv(path, sep="\t", comment="#", header=0)
                df["sample"] = sample
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Could not parse {path}: {e}")

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def load_wakhan_cnv_all(root_folder: str, samples: list) -> pd.DataFrame:
        """Load Wakhan CNV for all samples and concatenate."""
        frames = [
            CNVmethods.load_wakhan_cnv(root_folder, s)
            for s in samples
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print("[WARN] No Wakhan CNV data loaded for any sample.")
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def plot_cnv_around_gene(
        cnv_all: pd.DataFrame,
        gene: "str | dict",
        samples: list,
        flank_bp: int = 500_000,
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        value_col: str = "copy_number",
    ):
        """
        Plot CNV segments around a gene locus for each sample.

        NOTE: chrom_col/start_col/end_col/value_col are placeholders — adjust once
        Wakhan output format is confirmed.
        """
        if isinstance(gene, str):
            coords = GENE_COORDS[gene]
            gene_name = gene
        else:
            coords = gene
            gene_name = f"chr{coords['chrom']}:{coords['start']}-{coords['end']}"

        gene_chr   = coords["chrom"]
        gene_start = coords["start"]
        gene_end   = coords["end"]
        win_start  = gene_start - flank_bp
        win_end    = gene_end   + flank_bp

        if cnv_all is None or cnv_all.empty:
            print(f"[INFO] No CNV data available yet — Wakhan results pending for {gene_name}.")
            return

        def _norm(c):
            return str(c).replace("chr", "").strip()

        mask = (
            cnv_all[chrom_col].map(_norm).eq(gene_chr)
            & (cnv_all[end_col]   >= win_start)
            & (cnv_all[start_col] <= win_end)
        )
        df_win = cnv_all.loc[mask].copy()

        if df_win.empty:
            print(f"[INFO] No CNV segments in {gene_name} window.")
            return

        n_samples = len(samples)
        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 3), sharey=True)
        if n_samples == 1:
            axes = [axes]

        for ax, sample in zip(axes, samples):
            sub = df_win.loc[df_win["sample"] == sample] if "sample" in df_win.columns else df_win
            ax.set_xlim(win_start, win_end)
            ax.set_title(sample, fontsize=_fs(9))
            ax.axvspan(gene_start, gene_end, color="grey", alpha=0.2)
            ax.axvline(gene_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(gene_end,   color="grey", linestyle="--", linewidth=1)
            ax.axhline(2, color="black", linestyle=":", linewidth=0.8)  # diploid baseline

            if not sub.empty and value_col in sub.columns:
                for _, row in sub.iterrows():
                    s = max(row[start_col], win_start)
                    e = min(row[end_col],   win_end)
                    v = row[value_col]
                    ax.hlines(v, s, e, linewidth=3)

            ax.set_xlabel(f"chr{gene_chr} position (bp)")

        axes[0].set_ylabel(value_col)
        fig.suptitle(f"CNV around {gene_name} (Wakhan)", y=1.01)
        plt.tight_layout()
        plt.show()

##########################################################
#### COHORT / MULTI-SAMPLE (MULTI-PATIENT) ANALYSIS    ####
##########################################################

# hg38 primary chromosome lengths — used to lay out genome-wide cohort plots.

# hg38 centromere spans (UCSC cytoBand "acen", merged, rounded to 100 kb).
# Approximate on purpose — they are used to blank unmappable bins, where a
# megabase either way costs nothing. Replace from a cytoBand file if you need
# exact boundaries for anything else.
# Plot output folders. True keeps every run in its own timestamped directory;
# False writes into the base path and overwrites.
#
# Set it either way:
#     os.environ["VA_NEW_PLOT_FOLDER"] = "0"          # survives a reload
#     variations_analyzer.NEW_FOLDER_PER_RUN = False  # reset by a reload
#
# The environment variable exists because the notebook calls
# importlib.reload(variations_analyzer) partway down, which restores this
# module global to its default and silently undid the setting.
NEW_FOLDER_PER_RUN = True

# Every figure's font sizes are multiplied by this. They are hardcoded per
# element (a tick label should stay smaller than a title), so a global
# rcParams change does not reach them — this scales them together instead.
#     import variations_analyzer as va; va.FONT_SCALE = 1.5
FONT_SCALE = 1.60


def _fs(size):
    """Scale a hardcoded font size by the module-level FONT_SCALE."""
    return size * FONT_SCALE


# The oncoprint legend is deliberately NOT scaled by FONT_SCALE. Raising
# FONT_SCALE is how the sample labels are made readable at poster size, but the
# legend is already the tallest thing in the top-right cell — scaling it too
# runs it down over the "% cohort" bars. 9.1 pt is 7 * the original 1.30 scale,
# i.e. the size the legend had before FONT_SCALE became a knob. Override per
# call with legend_fontsize=.
ONCOPRINT_LEGEND_PT = 9.1


def _new_folder_per_run() -> bool:
    env = os.environ.get("VA_NEW_PLOT_FOLDER")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off")
    return bool(NEW_FOLDER_PER_RUN)

HG38_CENTROMERES = {
    "1":  (121_700_000, 125_100_000), "2":  (91_800_000,  96_000_000),
    "3":  (87_800_000,  94_000_000),  "4":  (48_200_000,  51_800_000),
    "5":  (46_100_000,  50_100_000),  "6":  (58_500_000,  62_600_000),
    "7":  (58_100_000,  62_100_000),  "8":  (43_200_000,  47_200_000),
    "9":  (42_200_000,  45_500_000),  "10": (38_000_000,  41_600_000),
    "11": (51_000_000,  55_800_000),  "12": (33_200_000,  37_800_000),
    "13": (16_500_000,  18_900_000),  "14": (16_100_000,  18_200_000),
    "15": (17_500_000,  20_500_000),  "16": (35_300_000,  38_300_000),
    "17": (22_700_000,  27_400_000),  "18": (15_400_000,  21_000_000),
    "19": (24_400_000,  28_100_000),  "20": (25_700_000,  30_400_000),
    "21": (10_900_000,  13_000_000),  "22": (13_700_000,  17_400_000),
    "X":  (58_100_000,  63_800_000),  "Y":  (10_300_000,  10_600_000),
}

# Large heterochromatic / repeat blocks that are not centromeres but behave the
# same way for a segment caller: it emits both a gain and a loss over them in
# most samples. Masking only the acen band left 100% spikes at 1q12, 9q12,
# 16q11.2 and on every acrocentric p-arm, which is what these cover.
def is_masked_region(chrom, start, end, telomere_bp=2_000_000,
                     centromere_pad=2_000_000):
    """
    True if [start, end) overlaps a centromere, telomere or heterochromatic /
    segdup block — regions where segment callers emit gains and losses that are
    mappability, not biology. Shared by every genome-scale plot so they agree
    about what is off-limits.
    """
    c = str(chrom)
    c = c[3:] if c.lower().startswith("chr") else c
    size = HG38_CHROM_SIZES.get(c)
    if size is None:
        return False
    if telomere_bp and (end <= telomere_bp or start >= size - telomere_bp):
        return True
    if c in HG38_CENTROMERES:
        a, b = HG38_CENTROMERES[c]
        if not (end < a - centromere_pad or start > b + centromere_pad):
            return True
    return any(not (end < a or start > b)
               for a, b in HG38_HETEROCHROMATIN.get(c, []))


def mask_subtract(chrom, start, end, telomere_bp=2_000_000,
                  centromere_pad=2_000_000):
    """
    [start, end) with every masked region removed, as a list of pieces.

    Segment callers emit arm-length segments, so testing a whole segment for
    *overlap* and dropping it deletes the entire arm — which is what happened to
    the patient tracks. Subtracting instead keeps the mappable parts and only
    blanks the block itself.
    """
    c = str(chrom)
    c = c[3:] if c.lower().startswith("chr") else c
    size = HG38_CHROM_SIZES.get(c)
    if size is None:
        return [(start, end)]

    blocks = []
    if telomere_bp:
        blocks += [(0, telomere_bp), (size - telomere_bp, size)]
    if c in HG38_CENTROMERES:
        a, b = HG38_CENTROMERES[c]
        blocks.append((a - centromere_pad, b + centromere_pad))
    blocks += list(HG38_HETEROCHROMATIN.get(c, []))

    pieces = [(start, end)]
    for a, b in sorted(blocks):
        out = []
        for ps, pe in pieces:
            if pe <= a or ps >= b:          # untouched
                out.append((ps, pe))
                continue
            if ps < a:
                out.append((ps, min(pe, a)))
            if pe > b:
                out.append((max(ps, b), pe))
        pieces = out
    return [(int(a), int(b)) for a, b in pieces if b > a]


HG38_HETEROCHROMATIN = {
    "1":  [(125_100_000, 143_200_000),    # 1q12
           (143_200_000, 152_500_000)],   # 1q21.1-q21.2 segdup cluster
    "9":  [(38_000_000,  66_000_000)],    # 9p11-q12: the largest such block in
                                          # the genome; a narrower span left
                                          # 100% loss bars at 39 and 61-66 Mb
    "16": [(31_500_000,  47_500_000)],    # 16p11.2-q11.2: segdup-rich, and
                                          # called as a 98% loss end to end
    "13": [(0, 18_900_000)], "14": [(0, 18_200_000)],   # acrocentric p-arms
    "15": [(0, 20_500_000)], "21": [(0, 13_000_000)],
    "22": [(0, 17_400_000)],
    "Y":  [(10_600_000, 57_300_000)],
}

HG38_CHROM_SIZES: dict[str, int] = {
    "1": 248_956_422, "2": 242_193_529, "3": 198_295_559, "4": 190_214_555,
    "5": 181_538_259, "6": 170_805_979, "7": 159_345_973, "8": 145_138_636,
    "9": 138_394_717, "10": 133_797_422, "11": 135_086_622, "12": 133_275_309,
    "13": 114_364_328, "14": 107_043_718, "15": 101_991_189, "16":  90_338_345,
    "17":  83_257_441, "18":  80_373_285, "19":  58_617_616, "20":  64_444_167,
    "21":  46_709_983, "22":  50_818_468, "X": 156_040_895, "Y":  57_227_415,
}

# hg38 gene bodies, verified 2026-08-27 against the UCSC hg38 HGNC track and
# Ensembl GRCh38. Every entry is the full HGNC gene span (not the MANE
# transcript span, which is shorter for several of these).
# NOTE resolve_gene() searches GENE_COORDS first, so TBXT / FN1 / SOX9 are
# served from that dict — the copies here must stay in step with it.
# hg38 gene spans for the chordoma panel (GENCODE v44, verified 2026-08-27).
CHORDOMA_GENE_COORDS: dict[str, dict] = {
    "TBXT":    {"chrom": "6",  "start": 166_157_656, "end": 166_168_700},
    "FN1":     {"chrom": "2",  "start": 215_360_440, "end": 215_436_073},
    "SOX9":    {"chrom": "17", "start":  72_121_020, "end":  72_126_416},
    "CDKN2A":  {"chrom": "9",  "start":  21_967_752, "end":  21_995_301},
    "CDKN2B":  {"chrom": "9",  "start":  22_002_903, "end":  22_009_305},
    # The two sit 7.6 kb apart and are usually taken out by a single deletion,
    # so the combined locus is often what you want to test. Span = CDKN2A start
    # to CDKN2B end, identical to the verified CDKN2A/B entry in gene_coords.py.
    "CDKN2A/B": {"chrom": "9", "start":  21_967_752, "end":  22_009_305},
    "PIK3CA":  {"chrom": "3",  "start": 179_148_114, "end": 179_240_093},
    "PTEN":    {"chrom": "10", "start":  87_862_638, "end":  87_971_930},
    "PBRM1":   {"chrom": "3",  "start":  52_545_352, "end":  52_685_917},
    "SETD2":   {"chrom": "3",  "start":  47_016_428, "end":  47_164_113},
    "LYST":    {"chrom": "1",  "start": 235_661_041, "end": 235_883_724},
    "TP53":    {"chrom": "17", "start":   7_661_779, "end":   7_687_538},
    "NF1":     {"chrom": "17", "start":  31_094_927, "end":  31_382_116},
    "SMARCB1": {"chrom": "22", "start":  23_786_931, "end":  23_838_009},
    "EGFR":    {"chrom": "7",  "start":  55_019_017, "end":  55_211_628},
    "TERT":    {"chrom": "5",  "start":   1_253_147, "end":   1_295_068},
}

# Default oncoprint row order for chordoma: TBXT first (brachyury, the defining
# locus), then the recurrently altered tumour-suppressor / PI3K genes.
CHORDOMA_PANEL: list[str] = [
    "TBXT", "CDKN2A", "CDKN2B", "PIK3CA", "PTEN", "PBRM1", "SETD2",
    "LYST", "TP53", "NF1", "SMARCB1", "FN1", "SOX9", "EGFR",
]

# Colors for the oncoprint alteration layers
ALT_COLORS: dict[str, str] = {
    "SNV":      "#4da6ff",   # light blue   — substitution
    "INDEL":    "#003f8a",   # navy         — insertion / deletion
    # CNV is drawn as a full-cell background with the SV bar on top, so these
    # stay light. They also match SNVmethods.plot_oncoprint's palette, and must
    # not collide with the saturated SVTYPE_COLORS (DEL is #d62728 red).
    "AMP":      "#ffbcaf",   # salmon       — copy gain
    "LOSS":     "#a5d5d8",   # teal         — copy loss
    "ABSENT":   "#ebebeb",   # light grey   — no alteration
    "NODATA":   "#f7f7f7",   # near white   — layer unavailable
}


class CohortMethods:
    """
    Cohort-level (multi-sample / multi-patient) integration of SNV, indel, SV,
    CNV and methylation data.

    Loaders (genome-wide, one row per event)
    ----------------------------------------
    · load_snv_cohort              — ClairS-TO snv+indel VCFs, PASS only
    · load_sv_cohort               — SV caller VCFs (sniffles / severus / ...)
    · load_coral_amplicon_segments — CORAL `*_graph.txt` amplicon segments (focal CN)
    · load_methylation_cohort      — modkit bedMethyl -> region x sample beta matrix

    Cohort plots
    ------------
    · plot_cohort_oncoprint        — gene x sample oncoprint, SNV/INDEL/SV/CNV layered
    · plot_cohort_burden           — per-sample SNV/indel burden + SV burden by type
    · plot_cnv_frequency_genome    — GISTIC-style gain/loss frequency across the genome
    · plot_sv_breakpoint_recurrence— fraction of the cohort with a breakpoint per bin
    · plot_cohort_chrom_heatmap    — sample x chromosome event-count heatmap
    · plot_methylation_umap        — UMAP / t-SNE embedding of methylation profiles
    · plot_methylation_heatmap     — hierarchically clustered beta-value heatmap

    Every plot accepts `save_dir`; a datetime-stamped folder is created the same
    way as in SNVmethods (shared per session via `_gene_save_dir`).
    """

    # ── small helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _norm_chrom(chrom: str) -> str:
        return str(chrom).replace("chr", "").replace("Chr", "").strip()

    @staticmethod
    def _cache_read(path: str) -> "pd.DataFrame | None":
        """
        Read a cached table. Parquet is used when pyarrow/fastparquet is
        installed; otherwise (and for .pkl paths) pandas pickle is used, which
        needs no extra dependency.
        """
        try:
            if str(path).endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_pickle(path)
        except (ImportError, OSError, ValueError) as e:
            print(f"[WARN] Could not read cache {path}: {e}")
            return None

    @staticmethod
    def _cache_write(df: pd.DataFrame, path: str) -> None:
        """Write a cached table, falling back to pickle when parquet is unavailable."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            if str(path).endswith(".parquet"):
                df.to_parquet(path, index=False)
            else:
                df.to_pickle(path)
            print(f"Cached to {path}")
        except ImportError:
            alt = os.path.splitext(path)[0] + ".pkl"
            df.to_pickle(alt)
            print(f"[INFO] parquet engine not installed — cached to {alt} instead")

    @staticmethod
    def resolve_gene(gene: "str | dict") -> tuple[str, dict]:
        """
        Resolve a gene name to (name, {chrom,start,end}).
        Looks in GENE_COORDS first, then CHORDOMA_GENE_COORDS.
        """
        if not isinstance(gene, str):
            coords = gene
            return f"chr{coords['chrom']}:{coords['start']}-{coords['end']}", coords
        if gene in GENE_COORDS:
            return gene, GENE_COORDS[gene]
        if gene in CHORDOMA_GENE_COORDS:
            return gene, CHORDOMA_GENE_COORDS[gene]
        raise KeyError(
            f"Gene {gene!r} not found in GENE_COORDS or CHORDOMA_GENE_COORDS. "
            f"Pass a dict {{'chrom':..,'start':..,'end':..}} instead."
        )

    # ── cohort naming: ONTWGS ids -> Cohort<N>-P<k> ──────────────────────────

    _SEQ_RE = re.compile(r'^ONTWGS(?P<run>\d+)-(?P<seq>\d+)-(?P<acc>\d+)-NP\d+$')

    @staticmethod
    def parse_sample_seq(sample: str) -> "int | None":
        """Run-relative index from an ONTWGS id (ONTWGS12-35-240548-NP01 -> 35)."""
        m = CohortMethods._SEQ_RE.match(str(sample))
        return int(m.group("seq")) if m else None

    @staticmethod
    def _cohort_slug(cohort: str) -> str:
        """'Cohort1' -> 'c1'  (short form used on axes; titles spell it out)."""
        m = re.match(r'^\s*cohort\s*[-_ ]?(\d+)\s*$', str(cohort), re.I)
        return f"c{m.group(1)}" if m else str(cohort).strip().lower().replace(" ", "")

    # Matches both label schemes: the old cohort form "c2-p14" and the patient
    # form "p32-s1" / "p7". Without the optional tail, "p7" fell through to the
    # string-sorted bucket and columns came out p1, p10, p11 … p2, p20.
    _LABEL_RE = re.compile(
        r'^(?P<c>[A-Za-z]+)(?P<ci>\d+)(?:-(?P<p>[A-Za-z]+)(?P<pi>\d+))?$', re.I)

    @staticmethod
    def sort_samples(samples) -> list:
        """
        Natural cohort/patient order: c1-p1, c1-p2 … c1-p10 … c2-p1 …
        (plain string sort would put c1-p10 before c1-p2). Labels that don't
        match the pattern are appended in string order.
        """
        matched, other = [], []
        for s in samples:
            m = CohortMethods._LABEL_RE.match(str(s))
            (matched if m else other).append(s)
        def _key(s):
            m = CohortMethods._LABEL_RE.match(str(s))
            return (m.group("c").lower(), int(m.group("ci")),
                    int(m.group("pi") or 0))

        matched.sort(key=_key)
        return matched + sorted(other, key=str)

    @staticmethod
    def build_sample_map(cohorts: dict) -> tuple[dict, dict]:
        """
        Map raw ONTWGS sample ids onto short cohort/patient labels.

            cohorts = {"Cohort1": samples_ontwgs10, "Cohort2": samples_ontwgs12}
            sample_map, cohort_of = CohortMethods.build_sample_map(cohorts)
            # sample_map["ONTWGS12-35-240548-NP01"] -> "c2-p1"

        Labels are lower-case and short (`c2-p1`) because they end up as tick
        labels on 49-column figures; plot titles and legends spell out
        "Cohort2" in full.

        Numbering restarts at 1 in every cohort and follows the **numeric**
        run-relative index, so ONTWGS12-35 → p1, -36 → p2 … -78 → p44
        regardless of how the directory listing sorted. Ids that don't match the
        ONTWGS pattern are appended after the numbered ones in string order, so
        nothing is silently dropped.

        Returns
        -------
        sample_map : {raw_id -> "c<N>-p<k>"}
        cohort_of  : {"c<N>-p<k>" -> "Cohort<N>"}  — full name, ready to hand to
                     plot_cohort_oncoprint(annotations={"Cohort": cohort_of})
        """
        sample_map, cohort_of = {}, {}
        for cohort, samples in cohorts.items():
            slug = CohortMethods._cohort_slug(cohort)
            with_seq, without = [], []
            for s in samples:
                (with_seq if CohortMethods.parse_sample_seq(s) is not None
                 else without).append(s)
            with_seq.sort(key=CohortMethods.parse_sample_seq)
            without.sort()
            for i, s in enumerate(with_seq + without, start=1):
                label = f"{slug}-p{i}"
                sample_map[s] = label
                cohort_of[label] = cohort
            if without:
                print(f"[WARN] {cohort}: {len(without)} id(s) without an ONTWGS "
                      f"index were numbered last: {', '.join(without[:3])}")
            if with_seq:
                first, last = with_seq[0], with_seq[-1]
                print(f"{cohort}: {len(with_seq) + len(without)} samples — "
                      f"{first} → {sample_map[first]} … {last} → {sample_map[last]}")
        return sample_map, cohort_of

    @staticmethod
    def relabel_samples(df: pd.DataFrame, sample_map: dict,
                        cohort_of: "dict | None" = None,
                        col: str = "sample") -> pd.DataFrame:
        """
        Rewrite `col` from raw ONTWGS ids to Cohort-P labels and (when
        `cohort_of` is given) add a `cohort` column. Rows whose sample is not in
        the map are dropped, with a warning — that is the intended behaviour
        when a directory holds samples outside the cohort of interest.
        """
        if df is None or df.empty:
            return df
        known = df[col].isin(sample_map)
        if not known.all():
            dropped = sorted(df.loc[~known, col].unique())
            print(f"[WARN] Dropping {(~known).sum():,} rows from "
                  f"{len(dropped)} unmapped sample(s): {', '.join(dropped[:3])}"
                  f"{' …' if len(dropped) > 3 else ''}")
        out = df.loc[known].copy()
        out[col] = out[col].map(sample_map)
        if cohort_of is not None:
            out["cohort"] = out[col].map(cohort_of)
        return out

    @staticmethod
    def gene_coords_from_gtf(gtf_path: str, genes: "list[str]") -> dict:
        """
        Pull hg38 gene-body coordinates for `genes` out of a GENCODE GTF
        (plain or .gz). Use this to refresh CHORDOMA_GENE_COORDS if you move to
        a different annotation release.

        Returns {gene: {"chrom":..,"start":..,"end":..}} for protein-coding
        entries on primary contigs.
        """
        want   = set(genes)
        opener = gzip.open if str(gtf_path).endswith(".gz") else open
        name_re = re.compile(r'gene_name "([^"]+)"')
        type_re = re.compile(r'gene_type "([^"]+)"')

        out: dict[str, dict] = {}
        found_type: dict[str, str] = {}
        with opener(gtf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.split("\t")
                if len(f) < 9 or f[2] != "gene":
                    continue
                m = name_re.search(f[8])
                if m is None or m.group(1) not in want:
                    continue
                chrom = CohortMethods._norm_chrom(f[0])
                if chrom not in PRIMARY_CHROMS:
                    continue
                name  = m.group(1)
                gtype = type_re.search(f[8])
                gtype = gtype.group(1) if gtype else "unknown"
                # prefer the protein-coding entry when a name is duplicated
                if name in out and found_type[name] == "protein_coding" and gtype != "protein_coding":
                    continue
                out[name] = {"chrom": chrom, "start": int(f[3]), "end": int(f[4])}
                found_type[name] = gtype

        missing = sorted(want - set(out))
        if missing:
            print(f"[WARN] Not found in {os.path.basename(gtf_path)}: {', '.join(missing)}")
        return out

    # ── genome-wide SNV / indel loading ──────────────────────────────────────

    @staticmethod
    def _parse_clairs_vcf_stream(path: str, sample: str, vtype: str,
                                 min_vaf: "float | None", max_vaf: "float | None",
                                 pass_only: bool = True) -> pd.DataFrame:
        """
        Stream a bgzipped ClairS-TO VCF and return PASS calls as a DataFrame.

        Reads the file line by line instead of going through vcfpy — for
        genome-wide VCFs (tens of millions of records, most of them non-PASS)
        this is roughly an order of magnitude faster.

        FORMAT is GT:GQ:DP:AF:AD:...  — DP and AF are looked up by name so a
        different field order still works.
        """
        rows = []
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line[0] == "#":
                        continue
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 10:
                        continue
                    if pass_only and f[6] != "PASS":
                        continue
                    chrom = CohortMethods._norm_chrom(f[0])
                    if chrom not in PRIMARY_CHROMS:
                        continue

                    keys = f[8].split(":")
                    vals = f[9].split(":")
                    fmt  = dict(zip(keys, vals))
                    try:
                        vaf = float(fmt.get("AF", "nan"))
                    except ValueError:
                        vaf = float("nan")
                    try:
                        dp = float(fmt.get("DP", "nan"))
                    except ValueError:
                        dp = float("nan")

                    if min_vaf is not None and not (vaf >= min_vaf):
                        continue
                    if max_vaf is not None and not (vaf <= max_vaf):
                        continue

                    ref, alt = f[3], f[4]
                    rows.append((sample, chrom, int(f[1]), ref, alt, vaf, dp,
                                 "SNV" if (len(ref) == 1 and len(alt) == 1) else "INDEL"))
        except OSError as e:
            print(f"[WARN] Could not read {path}: {e}")

        return pd.DataFrame(
            rows,
            columns=["sample", "chrom", "pos", "ref", "alt", "vaf", "dp", "vtype"],
        )

    @staticmethod
    def _restrict_cached(df, samples, cache_path):
        """
        Drop rows whose sample is not in the requested list.

        Caches are keyed only by path, so one written when the cohort was wider
        silently reintroduces samples that have since been excluded. Filtering
        on read makes the cache follow the caller rather than the other way
        round.
        """
        if df is None or df.empty or not samples or "sample" not in df.columns:
            return df
        want = set(samples)
        extra = sorted(set(df["sample"].astype(str)) - want)
        if extra:
            print(f"[INFO] dropping {len(extra)} sample(s) from the cache at "
                  f"{os.path.basename(cache_path)}, not in the requested list: "
                  f"{', '.join(extra[:4])}{' …' if len(extra) > 4 else ''}")
            df = df.loc[df["sample"].isin(want)]
        return df

    @staticmethod
    def load_snv_cohort(
        clairsto_root: str,
        samples: list,
        include_indel: bool = True,
        min_vaf: "float | None" = 0.1,
        max_vaf: "float | None" = 0.6,
        pass_only: bool = True,
        n_jobs: int = 8,
        cache_path: "str | None" = None,
        snv_patterns: tuple = ("snv.vcf.gz", "*.clairs_to.snv.vcf.gz", "*snv.vcf.gz"),
        indel_patterns: tuple = ("indel.vcf.gz", "*.clairs_to.indel.vcf.gz", "*indel.vcf.gz"),
    ) -> pd.DataFrame:
        """
        Load genome-wide ClairS-TO calls for every sample in the cohort.

        Returns one row per variant:
            sample, chrom, pos, ref, alt, vaf, dp, vtype ("SNV" | "INDEL")

        The default VAF window (0.1–0.6) is the somatic range used elsewhere in
        this notebook; pass min_vaf=None, max_vaf=None to keep everything.

        VCFs are found under `{clairsto_root}/{sample}/` by trying each pattern
        in turn, which covers both layouts in use here:
          · ONTWGS12 cordoma — `snv.vcf.gz` / `indel.vcf.gz`
          · nf-ontwgs10 output — `{sample}.clairs_to.snv.vcf.gz` (no indel file)
        A cohort with no indel VCF simply contributes no INDEL rows.

        `cache_path` is written after the first load and reused afterwards — a
        full cohort pass decompresses several GB.
        """
        if cache_path and os.path.exists(cache_path):
            df = CohortMethods._cache_read(cache_path)
            if df is not None:
                # Re-apply the sample list. A cache written when the cohort was
                # wider would otherwise put dropped samples straight back in —
                # which is how excluded samples kept reappearing downstream.
                df = CohortMethods._restrict_cached(df, samples, cache_path)
                print(f"Loaded {len(df):,} cached variants from {cache_path}")
                return df

        def _first_match(sdir, patterns):
            for pat in patterns:
                hits = sorted(glob.glob(os.path.join(sdir, pat)))
                if hits:
                    return hits[0]
            return None

        jobs = []
        n_indel = 0
        for s in samples:
            sdir = os.path.join(clairsto_root, s)
            snv_vcf = _first_match(sdir, snv_patterns)
            if snv_vcf:
                jobs.append((snv_vcf, s, "SNV"))
            else:
                print(f"[WARN] No SNV VCF for {s} under {sdir}")
            if include_indel:
                indel_vcf = _first_match(sdir, indel_patterns)
                if indel_vcf:
                    jobs.append((indel_vcf, s, "INDEL"))
                    n_indel += 1
        if include_indel and n_indel == 0 and jobs:
            print("[INFO] No indel VCFs in this callset — SNV rows only.")

        if not jobs:
            print(f"[WARN] No ClairS-TO VCFs found under {clairsto_root}")
            return pd.DataFrame(columns=["sample", "chrom", "pos", "ref", "alt",
                                         "vaf", "dp", "vtype"])

        print(f"Reading {len(jobs)} VCFs for {len(samples)} samples (n_jobs={n_jobs})...")
        frames = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
            joblib.delayed(CohortMethods._parse_clairs_vcf_stream)(
                path, sample, vt, min_vaf, max_vaf, pass_only
            )
            for path, sample, vt in jobs
        )
        frames = [f for f in frames if not f.empty]
        if not frames:
            print("[WARN] No PASS variants loaded.")
            return pd.DataFrame(columns=["sample", "chrom", "pos", "ref", "alt",
                                         "vaf", "dp", "vtype"])

        df = pd.concat(frames, ignore_index=True)
        vaf_note = "all VAFs" if (min_vaf is None and max_vaf is None) else f"VAF {min_vaf}–{max_vaf}"
        print(f"Loaded {len(df):,} PASS variants across {df['sample'].nunique()} samples ({vaf_note})")

        if cache_path:
            CohortMethods._cache_write(df, cache_path)
        return df

    # ── genome-wide SV loading ───────────────────────────────────────────────

    @staticmethod
    def _parse_info(info: str) -> dict:
        out = {}
        for field in info.split(";"):
            if "=" in field:
                k, v = field.split("=", 1)
                out[k] = v
            else:
                out[field] = True
        return out

    @staticmethod
    def _parse_sv_vcf_stream(path: str, sample: str, caller: str,
                             min_sv_len: int, pass_only: bool) -> pd.DataFrame:
        """Stream an SV VCF (plain or bgzipped) into a tidy DataFrame."""
        rows = []
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as fh:
                for line in fh:
                    if line[0] == "#":
                        continue
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 8:
                        continue
                    if pass_only and f[6] not in ("PASS", "."):
                        continue
                    chrom = CohortMethods._norm_chrom(f[0])
                    if chrom not in PRIMARY_CHROMS:
                        continue

                    info   = CohortMethods._parse_info(f[7])
                    svtype = str(info.get("SVTYPE", "")).upper()
                    if not svtype:
                        continue

                    start = int(f[1])
                    try:
                        end = int(info.get("END", start))
                    except (TypeError, ValueError):
                        end = start

                    chrom2 = CohortMethods._norm_chrom(info.get("CHR2", chrom))
                    try:
                        svlen = abs(int(float(info.get("SVLEN", end - start))))
                    except (TypeError, ValueError):
                        svlen = abs(end - start)

                    # size filter: BNDs have no span, so they always pass
                    if svtype != "BND" and svlen < min_sv_len:
                        continue

                    try:
                        af = float(info.get("AF", "nan"))
                    except ValueError:
                        af = float("nan")
                    try:
                        support = int(info.get("RE", info.get("SUPPORT", 0)))
                    except (TypeError, ValueError):
                        support = 0

                    rows.append((sample, caller, chrom, start, end, chrom2,
                                 svtype, svlen, af, support))
        except OSError as e:
            print(f"[WARN] Could not read {path}: {e}")

        return pd.DataFrame(
            rows,
            columns=["sample", "caller", "chrom", "start", "end", "chrom2",
                     "svtype", "svlen", "af", "support"],
        )

    @staticmethod
    def filter_sv_min_len(
        sv: pd.DataFrame,
        min_len: int = 10_000,
        keep_interchrom_bnd: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Hard size floor applied to a loaded SV table, once, for every consumer.

        Use this when the upstream caller's own size cutoff did not take effect
        and re-running is not an option: filtering here makes the oncoprint, the
        burden plot, the recurrence track and the co-occurrence test all agree,
        instead of each applying (or forgetting) its own threshold.

        `svlen` decides it for every spanning type (DEL, DUP, INV, INS and the
        compound calls). BND carries no span — sniffles puts the mate coordinate
        in END — so:
          * different chromosomes -> a translocation, kept when
            `keep_interchrom_bnd` (there is no length to test);
          * same chromosome       -> kept only when the two breakends are at
            least `min_len` apart.

        Returns a filtered copy; the input is untouched.
        """
        if sv is None or sv.empty:
            return sv

        is_bnd = sv["svtype"].astype(str).str.upper().eq("BND")
        same_chrom = sv["chrom"].astype(str) == sv["chrom2"].astype(str)
        bnd_span = (sv["end"] - sv["start"]).abs()

        keep_spanning = ~is_bnd & (sv["svlen"].abs() >= min_len)
        keep_bnd = is_bnd & np.where(
            same_chrom, bnd_span >= min_len, bool(keep_interchrom_bnd))

        out = sv.loc[keep_spanning | keep_bnd].copy()

        if verbose:
            before, after = len(sv), len(out)
            print(f"SV size filter ≥{min_len:,} bp: {before:,} → {after:,} "
                  f"({after / max(before, 1):.1%} kept)")
            by_type = (pd.DataFrame({"before": sv["svtype"].value_counts(),
                                     "after": out["svtype"].value_counts()})
                         .fillna(0).astype(int))
            by_type["kept %"] = (by_type["after"] / by_type["before"].clip(lower=1)
                                 * 100).round(1)
            print(by_type.sort_values("before", ascending=False).to_string())
        return out

    @staticmethod
    def load_sv_cohort(
        sv_root: str,
        caller_folders: dict,
        samples: "list | None" = None,
        min_sv_len: int = 50,
        pass_only: bool = True,
        n_jobs: int = 8,
        cache_path: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Load genome-wide SV calls for the whole cohort.

        caller_folders : {"sniffles_v1": "ONTWGS12_cordoma_sniffles1", ...}
                         same mapping used by SVmethods.summarize_sv_counts.
        samples        : restrict to these sample IDs (None = every sample dir).

        Returns: sample, caller, chrom, start, end, chrom2, svtype, svlen, af, support
        """
        if cache_path and os.path.exists(cache_path):
            df = CohortMethods._cache_read(cache_path)
            if df is not None:
                df = CohortMethods._restrict_cached(df, samples, cache_path)
                print(f"Loaded {len(df):,} cached SVs from {cache_path}")
                return df

        keep = set(samples) if samples else None
        jobs = []
        for caller, subdir in caller_folders.items():
            caller_path = os.path.join(sv_root, subdir)
            if not os.path.isdir(caller_path):
                print(f"[WARN] Caller folder not found: {caller_path}")
                continue
            for sample_dir in sorted(SNVmethods._list_sample_dirs_sv(caller_path)):
                sample = os.path.basename(sample_dir.rstrip(os.sep))
                if keep is not None and sample not in keep:
                    continue
                vcfs = sorted(glob.glob(os.path.join(sample_dir, "*.vcf.gz")))
                if not vcfs:
                    vcfs = sorted(glob.glob(os.path.join(sample_dir, "*.vcf")))
                for vf in vcfs:
                    jobs.append((vf, sample, caller))

        if not jobs:
            print(f"[WARN] No SV VCFs found under {sv_root}")
            return pd.DataFrame(columns=["sample", "caller", "chrom", "start", "end",
                                         "chrom2", "svtype", "svlen", "af", "support"])

        print(f"Reading {len(jobs)} SV VCFs (n_jobs={n_jobs})...")
        frames = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
            joblib.delayed(CohortMethods._parse_sv_vcf_stream)(
                path, sample, caller, min_sv_len, pass_only
            )
            for path, sample, caller in jobs
        )
        frames = [f for f in frames if not f.empty]
        if not frames:
            print("[WARN] No SVs loaded.")
            return pd.DataFrame(columns=["sample", "caller", "chrom", "start", "end",
                                         "chrom2", "svtype", "svlen", "af", "support"])

        df = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(df):,} SVs across {df['sample'].nunique()} samples "
              f"({', '.join(sorted(df['caller'].unique()))}, min_sv_len={min_sv_len:,})")

        if cache_path:
            CohortMethods._cache_write(df, cache_path)
        return df

    # ── CORAL amplicon segments (focal copy number) ──────────────────────────

    _CORAL_SEG_RE = re.compile(
        r"^(?P<chrom>[^:]+):(?P<start>\d+)[+-]?\s+(?P<chrom2>[^:]+):(?P<end>\d+)[+-]?$"
    )

    @staticmethod
    def load_coral_amplicon_segments(coral_root: str, samples: list) -> pd.DataFrame:
        """
        Parse CORAL `*_amplicon<N>_graph.txt` files into a segment table.

        Each `sequence` line looks like:
            sequence  chr1:10469829-  chr1:11209727+  10.09  87.98  739899  100376
                      ^ start          ^ end           ^CN    ^cov   ^size  ^reads

        Returns: sample, amplicon, chrom, start, end, cn, coverage, size, n_reads

        NOTE: CORAL only reports segments inside amplified regions, so this is a
        focal-amplification view, not a genome-wide copy-number profile. Feed it
        to plot_cnv_frequency_genome to see recurrently amplified loci.
        """
        rows = []
        for sample in samples:
            sdir = os.path.join(coral_root, sample)
            if not os.path.isdir(sdir):
                print(f"[WARN] CORAL dir not found: {sdir}")
                continue
            for path in sorted(glob.glob(os.path.join(sdir, "*_graph.txt"))):
                m = re.search(r"amplicon(\d+)_graph\.txt$", os.path.basename(path))
                amplicon = int(m.group(1)) if m else -1
                try:
                    with open(path) as fh:
                        for line in fh:
                            if not line.startswith("sequence"):
                                continue
                            f = line.rstrip("\n").split("\t")
                            if len(f) < 5:
                                continue
                            sm = CohortMethods._CORAL_SEG_RE.match(f"{f[1].strip()} {f[2].strip()}")
                            if sm is None:
                                continue
                            chrom = CohortMethods._norm_chrom(sm.group("chrom"))
                            if chrom not in PRIMARY_CHROMS:
                                continue
                            start, end = int(sm.group("start")), int(sm.group("end"))
                            if end < start:
                                start, end = end, start
                            rows.append({
                                "sample":   sample,
                                "amplicon": amplicon,
                                "chrom":    chrom,
                                "start":    start,
                                "end":      end,
                                "cn":       float(f[3]),
                                "coverage": float(f[4]) if len(f) > 4 else float("nan"),
                                "size":     int(f[5]) if len(f) > 5 else end - start,
                                "n_reads":  int(f[6]) if len(f) > 6 else 0,
                            })
                except (OSError, ValueError) as e:
                    print(f"[WARN] Could not parse {path}: {e}")

        if not rows:
            print(f"[WARN] No CORAL amplicon segments found under {coral_root}")
            return pd.DataFrame(columns=["sample", "amplicon", "chrom", "start", "end",
                                         "cn", "coverage", "size", "n_reads"])
        df = pd.DataFrame(rows)
        print(f"Loaded {len(df):,} CORAL segments from "
              f"{df['sample'].nunique()} samples "
              f"({df.groupby('sample')['amplicon'].nunique().sum()} amplicons total)")
        return df

    # ── genome-wide copy number (Wakhan / CNVkit) ────────────────────────────

    @staticmethod
    def _read_wakhan_hp_bed(path: str) -> pd.DataFrame:
        """Read one Wakhan `*_copynumbers_segments_HP_<n>.bed` (comment header)."""
        rows = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 5:
                    continue
                chrom = CohortMethods._norm_chrom(f[0])
                if chrom not in PRIMARY_CHROMS:
                    continue
                try:
                    conf = float(f[5]) if len(f) > 5 else float("nan")
                    rows.append((chrom, int(f[1]), int(f[2]), float(f[3]),
                                 float(f[4]), conf))
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["chrom", "start", "end", "coverage",
                                           "cn_hp", "confidence"])

    @staticmethod
    def load_wakhan_cn_segments(wakhan_root: str, samples: list,
                                solution: "str | None" = None) -> pd.DataFrame:
        """
        Genome-wide total copy number from Wakhan haplotype-resolved output.

        Accepts either layout — the full nf output, or a slimmed-down copy with
        just the solution folders (all that is needed is ~20 kB per sample):
            {wakhan_root}/{sample}/wakhan_out/<ploidy>_<purity>_<conf>/bed_output/
            {wakhan_root}/{sample}/<ploidy>_<purity>_<conf>/bed_output/
                {sample}_<solution>_copynumbers_segments_HP_1.bed   (and HP_2)

        HP_1 and HP_2 carry per-haplotype copy-number states on their own
        breakpoints, so the two are combined on the union of breakpoints and
        summed — `cn` is the total copy number, with `cn_hp1`/`cn_hp2` kept for
        allele-specific work (LOH is cn_hp1 == 0 or cn_hp2 == 0).

        `solution` picks a specific <ploidy>_<purity>_<conf> folder; by default
        the single solution present is used.

        Returns: sample, chrom, start, end, cn, cn_hp1, cn_hp2, ploidy, purity
        """
        frames = []
        missing = []
        for sample in samples:
            # solution folders may sit under {sample}/wakhan_out/ (full nf output)
            # or directly under {sample}/ (slimmed-down copy)
            sol_dirs = sorted(
                d for parent in (os.path.join(wakhan_root, sample, "wakhan_out"),
                                 os.path.join(wakhan_root, sample))
                for d in glob.glob(os.path.join(parent, "*"))
                if os.path.isdir(os.path.join(d, "bed_output"))
            )
            if not sol_dirs:
                missing.append(sample)
                continue
            if solution:
                sol_dirs = [d for d in sol_dirs if os.path.basename(d) == solution]
            if not sol_dirs:
                print(f"[WARN] No Wakhan solution '{solution}' for {sample}")
                continue
            if len(sol_dirs) > 1:
                print(f"[INFO] {sample}: {len(sol_dirs)} Wakhan solutions, using "
                      f"{os.path.basename(sol_dirs[0])} (pass solution= to choose)")
            sol = sol_dirs[0]
            sol_name = os.path.basename(sol)

            hp = {}
            for n in (1, 2):
                hits = sorted(glob.glob(os.path.join(
                    sol, "bed_output", f"*copynumbers_segments_HP_{n}.bed")))
                if hits:
                    hp[n] = CohortMethods._read_wakhan_hp_bed(hits[0])
            if 1 not in hp or 2 not in hp:
                print(f"[WARN] {sample}: missing an HP copy-number bed — skipped")
                continue

            # Wakhan names each folder solution_<ploidy>_<purity>_<confidence>
            # and ranks solutions by that confidence. It is SOLUTION-level — how
            # good this purity/ploidy fit is against the alternatives — and is a
            # different quantity from the per-segment confidence the beds carry.
            # Measured here the two correlate at only 0.20 (0.32-0.48 vs
            # 0.63-0.96), so they must not be plotted on a shared axis.
            parts = sol_name.split("_")
            try:
                ploidy, purity = float(parts[0]), float(parts[1])
            except (IndexError, ValueError):
                ploidy = purity = float("nan")
            try:
                solution_score = float(parts[2])
            except (IndexError, ValueError):
                solution_score = float("nan")

            # Combine haplotypes on the union of their breakpoints. Wakhan writes
            # inclusive intervals (one segment ends at X, the next starts at X+1),
            # so treat each as half-open [start, end+1) and look CN up with a step
            # function — a plain containment test would leave 1 bp holes and NaNs.
            def _step_lookup(seg: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
                starts = seg["start"].to_numpy()
                ends   = seg["end"].to_numpy() + 1
                cns    = seg["cn_hp"].to_numpy()
                idx = np.searchsorted(starts, positions, side="right") - 1
                out = np.full(positions.shape, np.nan)
                ok = idx >= 0
                within = np.zeros_like(ok)
                within[ok] = positions[ok] < ends[idx[ok]]
                out[within] = cns[idx[within]]
                return out

            def _conf_at(frame, pos):
                """Per-segment confidence at a position, NaN outside any segment."""
                if frame.empty or "confidence" not in frame.columns:
                    return float("nan")
                m = frame.loc[(frame["start"] <= pos) & (pos <= frame["end"]),
                              "confidence"]
                return float(m.iloc[0]) if len(m) else float("nan")

            recs = []
            for chrom in set(hp[1]["chrom"]) | set(hp[2]["chrom"]):
                a = hp[1].loc[hp[1]["chrom"] == chrom].sort_values("start")
                b = hp[2].loc[hp[2]["chrom"] == chrom].sort_values("start")
                edges = np.unique(np.concatenate([
                    a["start"].to_numpy(), a["end"].to_numpy() + 1,
                    b["start"].to_numpy(), b["end"].to_numpy() + 1,
                ]))
                if edges.size < 2:
                    continue
                lo, hi = edges[:-1], edges[1:]
                mid = (lo + hi) / 2.0
                cn1 = _step_lookup(a, mid) if not a.empty else np.full(mid.shape, np.nan)
                cn2 = _step_lookup(b, mid) if not b.empty else np.full(mid.shape, np.nan)

                keep = ~(np.isnan(cn1) & np.isnan(cn2))
                lo, hi, cn1, cn2 = lo[keep], hi[keep], cn1[keep], cn2[keep]
                if lo.size == 0:
                    continue

                # collapse runs with identical haplotype states (removes slivers)
                same = np.concatenate([[False], (
                    (np.nan_to_num(cn1[1:], nan=-1) == np.nan_to_num(cn1[:-1], nan=-1))
                    & (np.nan_to_num(cn2[1:], nan=-1) == np.nan_to_num(cn2[:-1], nan=-1))
                    & (lo[1:] == hi[:-1])
                )])
                run = np.cumsum(~same)
                for r in np.unique(run):
                    m = run == r
                    s, e = int(lo[m][0]), int(hi[m][-1])
                    c1, c2 = cn1[m][0], cn2[m][0]
                    cf = np.nanmean([_conf_at(a, (s + e) / 2.0),
                                     _conf_at(b, (s + e) / 2.0)])
                    recs.append((sample, chrom, s, e,
                                 float(np.nansum([c1, c2])), c1, c2,
                                 ploidy, purity, solution_score, float(cf)))

            if recs:
                frames.append(pd.DataFrame(recs, columns=[
                    "sample", "chrom", "start", "end", "cn",
                    "cn_hp1", "cn_hp2", "ploidy", "purity",
                    "solution_score", "confidence"]))

        if missing:
            print(f"[WARN] No Wakhan solution folder for {len(missing)}/{len(samples)} "
                  f"samples — they count as zero: {', '.join(missing[:5])}"
                  f"{' …' if len(missing) > 5 else ''}")

        if not frames:
            print(f"[WARN] No Wakhan copy-number segments loaded from {wakhan_root}")
            return pd.DataFrame(columns=["sample", "chrom", "start", "end", "cn",
                                         "cn_hp1", "cn_hp2", "ploidy", "purity",
                                         "solution_score", "confidence"])
        df = pd.concat(frames, ignore_index=True)
        # length-weighted mean CN should track Wakhan's own ploidy estimate —
        # a large gap means the wrong solution folder was picked
        lw = df.assign(_len=df["end"] - df["start"])
        chk = lw.groupby("sample").apply(
            lambda g: pd.Series({
                "mean_cn": np.average(g["cn"], weights=g["_len"]),
                "ploidy":  g["ploidy"].iloc[0],
                "purity":  g["purity"].iloc[0],
            }), include_groups=False)
        print(f"Loaded {len(df):,} Wakhan CN segments from "
              f"{df['sample'].nunique()} samples")
        print(f"  length-weighted mean CN {chk['mean_cn'].min():.2f}–"
              f"{chk['mean_cn'].max():.2f} vs Wakhan ploidy "
              f"{chk['ploidy'].min():.2f}–{chk['ploidy'].max():.2f}, "
              f"purity {chk['purity'].min():.2f}–{chk['purity'].max():.2f}")
        return df

    @staticmethod
    def load_cnvkit_cns(cnvkit_root: str, samples: list,
                        ploidy: float = 2.0) -> pd.DataFrame:
        """
        Genome-wide copy number from CNVkit `.cns` segments, as a cross-check on
        Wakhan. `cn` is derived from log2 ratio: cn = ploidy * 2**log2.

        Expects {cnvkit_root}/{sample}/{sample}.cns (or any *.cns in that dir).
        Returns: sample, chrom, start, end, log2, cn, depth, probes
        """
        frames = []
        for sample in samples:
            hits = sorted(glob.glob(os.path.join(cnvkit_root, sample, "*.cns")))
            hits = [h for h in hits if not h.endswith((".bintest.cns", ".call.cns"))] or hits
            if not hits:
                print(f"[WARN] No .cns for {sample} under {cnvkit_root}")
                continue
            try:
                df = pd.read_csv(hits[0], sep="\t")
            except (OSError, ValueError) as e:
                print(f"[WARN] Could not parse {hits[0]}: {e}")
                continue
            df = df.rename(columns={"chromosome": "chrom"})
            df["chrom"] = df["chrom"].map(CohortMethods._norm_chrom)
            df = df.loc[df["chrom"].isin(PRIMARY_CHROMS)].copy()
            df["cn"] = ploidy * np.power(2.0, df["log2"])
            df["sample"] = sample
            keep = [c for c in ["sample", "chrom", "start", "end", "log2", "cn",
                                "depth", "probes"] if c in df.columns]
            frames.append(df[keep])

        if not frames:
            print(f"[WARN] No CNVkit segments loaded from {cnvkit_root}")
            return pd.DataFrame(columns=["sample", "chrom", "start", "end",
                                         "log2", "cn"])
        df = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(df):,} CNVkit segments from {df['sample'].nunique()} samples")
        return df

    # ── CNV source registry ──────────────────────────────────────────────────

    # Copy-number callers are kept apart on purpose. CNVkit reports a continuous
    # log2 ratio against a flat reference (no purity/ploidy correction), Wakhan
    # reports purity-corrected integer states per haplotype, and CORAL only
    # reports segments inside amplicons. Merging them into one column would
    # silently compare quantities that are not the same measurement.
    CNV_SOURCES: tuple = ("cnvkit", "wakhan", "coral_amplicon")

    @staticmethod
    def load_cnv_sources(
        cohorts: dict,
        samples_by_cohort: dict,
        sample_map: "dict | None" = None,
        cohort_of: "dict | None" = None,
        sources: "tuple | None" = None,
    ) -> dict:
        """
        Load every copy-number source for every cohort, keeping the sources
        separate.

            cnv = CohortMethods.load_cnv_sources(COHORTS, RAW_SAMPLES,
                                                 SAMPLE_MAP, COHORT_OF)
            cnv["cnvkit"]          # all cohorts that have CNVkit output
            cnv["wakhan"]          # all cohorts that have Wakhan output

        `cohorts` is the notebook's registry — each entry may carry the keys
        `cnvkit`, `wakhan`, `coral` (set to None when that caller has not been
        run for the cohort yet). A cohort missing a source simply contributes no
        rows to it; see cnv_coverage_table for an explicit per-cohort tally.

        Returns {source: DataFrame} — always contains a key for every requested
        source, empty DataFrame when nothing was found.
        """
        sources = sources or CohortMethods.CNV_SOURCES
        key_for = {"cnvkit": "cnvkit", "wakhan": "wakhan", "coral_amplicon": "coral"}
        loader  = {
            "cnvkit":         CohortMethods.load_cnvkit_cns,
            "wakhan":         CohortMethods.load_wakhan_cn_segments,
            "coral_amplicon": CohortMethods.load_coral_amplicon_segments,
        }

        out = {}
        for src in sources:
            parts = []
            for cohort, cfg in cohorts.items():
                root = cfg.get(key_for[src])
                samples = samples_by_cohort.get(cohort, [])
                if not root:
                    print(f"[INFO] {cohort}: no {src} configured — counts as zero.")
                    continue
                if not os.path.isdir(root):
                    print(f"[INFO] {cohort}: {src} path not present ({root}) "
                          f"— counts as zero.")
                    continue
                print(f"\n─── {cohort} / {src} ───")
                df = loader[src](root, samples)
                if df is None or df.empty:
                    continue
                if sample_map:
                    df = CohortMethods.relabel_samples(df, sample_map, cohort_of)
                if not df.empty:
                    df["cnv_source"] = src
                    parts.append(df)
            out[src] = (pd.concat(parts, ignore_index=True) if parts
                        else pd.DataFrame(columns=["sample", "chrom", "start",
                                                   "end", "cn", "cnv_source"]))
        return out

    @staticmethod
    def cnv_coverage_table(cnv_sources: dict, samples_by_cohort: dict,
                           sample_map: "dict | None" = None) -> pd.DataFrame:
        """
        Cohort x CNV-source table of how many samples actually have segments —
        missing combinations show as an explicit 0 rather than quietly vanishing
        from the figures.
        """
        rows = []
        for cohort, raw in samples_by_cohort.items():
            labels = {sample_map.get(s, s) for s in raw} if sample_map else set(raw)
            row = {"cohort": cohort, "n_samples": len(labels)}
            for src, df in cnv_sources.items():
                row[src] = (0 if df is None or df.empty
                            else df.loc[df["sample"].isin(labels), "sample"].nunique())
            rows.append(row)
        tbl = pd.DataFrame(rows).set_index("cohort")
        missing = [(c, s) for c in tbl.index for s in cnv_sources if tbl.loc[c, s] == 0]
        if missing:
            print("[INFO] No copy-number data (counted as zero) for: "
                  + ", ".join(f"{c}/{s}" for c, s in missing))
        return tbl

    # ── alteration matrix ────────────────────────────────────────────────────

    @staticmethod
    def build_cohort_alterations(
        genes: list,
        samples: list,
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        flank_bp: int = 0,
        snv_min_vaf: float = 0.05,
        sv_min_len: int = 10_000,
        sv_svtypes: "list | None" = None,
        sv_exclude_types: "list | tuple | None" = None,
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_col: str = "cn",
        cnv_baseline: str = "absolute",
    ) -> pd.DataFrame:
        """
        Collapse genome-wide cohort tables into one row per (gene, sample) with
        boolean columns for each alteration layer.

        Returns columns:
            gene, sample, SNV, INDEL, SV, AMP, LOSS, sv_type, n_snv, n_sv, max_cn

        `flank_bp` widens each gene window on both sides (0 = gene body only).

        `sv_min_len` (default 10 kb, matching build_sv_all_for_gene) drops the
        small INS/DEL that dominate a long-read SV callset — without it nearly
        every gene is "SV altered" in every sample and the oncoprint says
        nothing. BNDs have no span and are always kept. Restrict further with
        e.g. sv_svtypes=["DEL","DUP","INV","BND"], or the inverse
        sv_exclude_types=["INV"].

        `cnv_baseline` decides how cnv_gain / cnv_loss are read:
          "absolute" — plain copy-number thresholds (2.5 / 1.5). Right for
                       CNVkit, whose cn is derived from a log2 ratio against a
                       flat diploid reference.
          "ploidy"   — thresholds become *multiples of the sample's own
                       baseline* (so 2.5/1.5 → 1.25x / 0.75x of baseline),
                       taken from the `ploidy` column when present and a
                       length-weighted median otherwise. Use this for Wakhan:
                       its states are purity-corrected and these tumours run
                       from ploidy 1.63 to 3.61, so a fixed "cn ≤ 1.5 = loss"
                       would call most of a near-haploid genome deleted.
        """
        recs = []

        snv_ok = snv_all is not None and not snv_all.empty
        sv_ok  = sv_all is not None and not sv_all.empty
        cnv_ok = (cnv_all is not None and not cnv_all.empty
                  and {"chrom", "start", "end", cnv_col, "sample"}.issubset(cnv_all.columns))

        if snv_ok:
            snv = snv_all
            if snv_min_vaf and "vaf" in snv.columns:
                snv = snv.loc[snv["vaf"].fillna(0) >= snv_min_vaf]
            snv = snv.assign(_chr=snv["chrom"].map(CohortMethods._norm_chrom))
        if sv_ok:
            sv = sv_all
            if sv_svtypes:
                sv = sv.loc[sv["svtype"].isin(sv_svtypes)]
            if sv_exclude_types:
                # blacklist rather than whitelist, so a type the caller starts
                # emitting later is kept by default instead of silently dropped
                sv = sv.loc[~sv["svtype"].isin(sv_exclude_types)]
            if sv_min_len and "svlen" in sv.columns:
                before = len(sv)
                sv = sv.loc[(sv["svtype"] == "BND") | (sv["svlen"].abs() >= sv_min_len)]
                print(f"[INFO] SV size filter ≥{sv_min_len:,} bp (BND exempt): "
                      f"{before:,} → {len(sv):,} calls")
            sv = sv.assign(
                _chr=sv["chrom"].map(CohortMethods._norm_chrom),
                _chr2=(sv["chrom2"].map(CohortMethods._norm_chrom)
                       if "chrom2" in sv.columns
                       else sv["chrom"].map(CohortMethods._norm_chrom)),
            )
            sv_ok = not sv.empty
        if cnv_ok:
            cnv = cnv_all.assign(_chr=cnv_all["chrom"].map(CohortMethods._norm_chrom))
            # per-sample gain/loss thresholds
            gain_thr, loss_thr = CohortMethods._cnv_thresholds(
                cnv, cnv_col, cnv_gain, cnv_loss, cnv_baseline)

        for gene in genes:
            gene_name, coords = CohortMethods.resolve_gene(gene)
            gchr = CohortMethods._norm_chrom(coords["chrom"])
            wstart = coords["start"] - flank_bp
            wend   = coords["end"]   + flank_bp

            snv_hits: dict = {}
            indel_hits: dict = {}
            if snv_ok:
                win = snv.loc[(snv["_chr"] == gchr)
                              & (snv["pos"] >= wstart) & (snv["pos"] <= wend)]
                if not win.empty:
                    if "vtype" in win.columns:
                        vt = win["vtype"]
                    else:
                        vt = np.where(
                            (win["ref"].astype(str).str.len() == 1)
                            & (win["alt"].astype(str).str.len() == 1), "SNV", "INDEL")
                    snv_hits   = win.loc[np.asarray(vt) == "SNV",   "sample"].value_counts().to_dict()
                    indel_hits = win.loc[np.asarray(vt) == "INDEL", "sample"].value_counts().to_dict()

            sv_hits: dict = {}
            sv_dominant: dict = {}
            if sv_ok:
                # A BND is two point breakends on possibly different chromosomes:
                # (chrom, start) and (chrom2, end). Its END is the MATE's
                # coordinate, so [start, end] is NOT an interval on `chrom` —
                # treating it as one invents a span across everything in between
                # and marks every gene along the way as altered.
                is_bnd = sv["svtype"] == "BND"
                span = (~is_bnd) & (sv["_chr"] == gchr) \
                       & (sv["end"] >= wstart) & (sv["start"] <= wend)
                bnd_a = is_bnd & (sv["_chr"] == gchr) \
                        & (sv["start"] >= wstart) & (sv["start"] <= wend)
                bnd_b = is_bnd & (sv["_chr2"] == gchr) \
                        & (sv["end"] >= wstart) & (sv["end"] <= wend)
                win = sv.loc[span | bnd_a | bnd_b]
                if not win.empty:
                    sv_hits = win["sample"].value_counts().to_dict()
                    sv_dominant = (win.groupby("sample")["svtype"]
                                      .agg(lambda x: x.value_counts().idxmax())
                                      .to_dict())

            amp_hits: dict = {}
            loss_hits: dict = {}
            max_cn: dict = {}
            if cnv_ok:
                win = cnv.loc[(cnv["_chr"] == gchr)
                              & (cnv["end"] >= wstart) & (cnv["start"] <= wend)]
                if not win.empty:
                    agg = win.groupby("sample")[cnv_col].max()
                    max_cn = agg.to_dict()
                    amp_hits = {s: True for s, v in agg.items()
                                if v >= gain_thr.get(s, cnv_gain)}
                    lo = win.groupby("sample")[cnv_col].min()
                    loss_hits = {s: True for s, v in lo.items()
                                 if v <= loss_thr.get(s, cnv_loss)}

            for s in samples:
                recs.append({
                    "gene":    gene_name,
                    "sample":  s,
                    "SNV":     bool(snv_hits.get(s, 0)),
                    "INDEL":   bool(indel_hits.get(s, 0)),
                    "SV":      bool(sv_hits.get(s, 0)),
                    "AMP":     bool(amp_hits.get(s, False)),
                    "LOSS":    bool(loss_hits.get(s, False)),
                    "sv_type": sv_dominant.get(s),
                    "n_snv":   int(snv_hits.get(s, 0)),
                    "n_indel": int(indel_hits.get(s, 0)),
                    "n_sv":    int(sv_hits.get(s, 0)),
                    "max_cn":  max_cn.get(s, float("nan")),
                })

        alt = pd.DataFrame(recs)
        alt["altered"] = alt[["SNV", "INDEL", "SV", "AMP", "LOSS"]].any(axis=1)
        if not snv_ok:
            print("[INFO] No SNV table supplied — SNV/INDEL rows will be empty.")
        if not sv_ok:
            print("[INFO] No SV table supplied — SV rows will be empty.")
        if not cnv_ok:
            print("[INFO] No usable CNV table supplied — AMP/LOSS rows will be empty.")
        return alt

    @staticmethod
    def _memo_sort(alt: pd.DataFrame, genes: list, samples: list) -> list:
        """
        Classic oncoprint waterfall ordering: samples are sorted by a binary
        key built from the gene rows, most-frequent gene first, so mutually
        exclusive patterns line up as a staircase.
        """
        wide = (alt.pivot_table(index="gene", columns="sample",
                                values="altered", aggfunc="any")
                   .reindex(index=genes, columns=samples)
                   .fillna(False))

        def key(sample):
            bits = wide[sample].to_numpy()
            score = 0
            for b in bits:                       # first gene is most significant bit
                score = (score << 1) | int(bool(b))
            return (-score, sample)

        return sorted(samples, key=key)

    # ── the oncoprint ────────────────────────────────────────────────────────

    @staticmethod
    def plot_cohort_oncoprint(
        alt: pd.DataFrame,
        samples: "list | None" = None,
        genes: "list | None" = None,
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        annotations: "dict | None" = None,
        annotation_order: "dict | None" = None,
        sample_labels: "dict | None" = None,
        label_rotation: int = 90,
        show_snv: bool = True,
        sort: str = "patient",
        title: str = "Chordoma cohort — alteration landscape",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
        figsize: "tuple | None" = None,
        xtick_scale: float = 1.0,
        legend_fontsize: "float | None" = None,
    ):
        """
        Cohort oncoprint. Columns = samples, rows = genes; within each cell the
        alteration layers are drawn on top of each other:

            full-height block   copy-number gain (red) / loss (blue)
            wide middle bar     SV, coloured by dominant SVTYPE
            thin inner bar(s)   SNV (light blue) / indel (navy)

        Parameters
        ----------
        alt         : output of build_cohort_alterations
        snv_all     : genome-wide SNV table — drives the top TMB bar (optional)
        sv_all      : genome-wide SV table  — drives the top SV-burden bar (optional)
        xtick_scale : multiplies ONLY the sample labels along the bottom, on top
                      of FONT_SCALE. The column labels are the first thing that
                      stops being readable when the figure is scaled down onto a
                      poster, and they are also the only text whose size is
                      bounded by column width rather than by taste, so they get
                      their own knob instead of being swept up in a global bump.
        legend_fontsize : absolute point size for the legend. Defaults to
                      ONCOPRINT_LEGEND_PT, which is fixed rather than scaled by
                      FONT_SCALE — see the note there.
        annotations : {"Methylation cluster": {sample: label}, "Site": {...}}
                      drawn as categorical colour strips under the grid, in the
                      order given. A sample missing from a mapping (or mapped to
                      None) draws grey — "no data", which is not the same as a
                      category and must not be confused with one.
        annotation_order : {name: [level, ...]} to fix the level order of a
                      strip. Numeric-looking levels (e.g. recurrence counts) are
                      sorted numerically and given a sequential palette
                      automatically, so low→high reads as light→dark.
        show_snv    : draw the thin SNV/indel inner bars. False leaves the CNV
                      background and the SV bar only — the structural view.
        sort        : "patient" (cohort then patient number), "memo"
                      (waterfall), "burden" (most altered first), or anything
                      else to keep `samples` exactly as given — which is how
                      PatientMethods.order_samples output is passed in.
        filename    : output basename; defaults to cohort_oncoprint_<sort>.pdf
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        genes   = genes   or list(dict.fromkeys(alt["gene"]))
        samples = samples or list(dict.fromkeys(alt["sample"]))

        if sort == "memo":
            samples = CohortMethods._memo_sort(alt, genes, samples)
        elif sort == "burden":
            burden = alt.groupby("sample")["altered"].sum()
            samples = sorted(samples, key=lambda s: (-burden.get(s, 0), s))
        elif sort == "patient":
            samples = CohortMethods.sort_samples(samples)

        idx = {(g, s): r for (g, s), r in
               alt.set_index(["gene", "sample"]).iterrows()}

        n_rows, n_cols = len(genes), len(samples)
        ann_names = list(annotations.keys()) if annotations else []
        n_ann = len(ann_names)

        cell_w = max(0.16, min(0.5, 16.0 / max(n_cols, 1)))
        fig_w  = figsize[0] if figsize else max(11.0, n_cols * cell_w + 3.4)
        fig_h  = figsize[1] if figsize else n_rows * 0.34 + n_ann * 0.30 + 4.2

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            3 if n_ann else 2, 2,
            height_ratios=([2.0, n_rows * 0.5, max(n_ann, 1) * 0.32] if n_ann
                           else [2.0, n_rows * 0.5]),
            width_ratios=[9.0, 0.85],   # the % panel only needs to be
                                        # readable, not to compete with
                                        # the grid for width
            hspace=0.06, wspace=0.03,
        )
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
        # deliberately NOT sharey with ax_main — clearing ticks on a shared axis
        # would wipe the gene labels off the oncoprint itself
        ax_right = fig.add_subplot(gs[1, 1])
        ax_ann   = fig.add_subplot(gs[2, 0], sharex=ax_top) if n_ann else None
        # Blank cell beside the burden panel — the legend lives here, directly
        # above the % cohort bars, instead of hanging off the figure's right
        # edge where it stretched the saved bbox and sat far from the grid.
        ax_leg   = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        # ── top: mutation burden ────────────────────────────────────────────
        x = np.arange(n_cols)
        if snv_all is not None and not snv_all.empty:
            if "vtype" in snv_all.columns:
                bt = (snv_all.groupby(["sample", "vtype"]).size()
                             .unstack(fill_value=0).reindex(samples, fill_value=0))
            else:
                bt = pd.DataFrame(index=samples)
            n_snv   = bt["SNV"].to_numpy()   if "SNV"   in bt.columns else np.zeros(n_cols)
            n_indel = bt["INDEL"].to_numpy() if "INDEL" in bt.columns else np.zeros(n_cols)
            ax_top.bar(x, n_snv,   width=0.82, color=ALT_COLORS["SNV"],   label="SNV")
            ax_top.bar(x, n_indel, width=0.82, bottom=n_snv,
                       color=ALT_COLORS["INDEL"], label="indel")
            ax_top.set_ylabel("PASS calls\n(genome-wide)", fontsize=_fs(8))
            ax_top.legend(fontsize=_fs(7), frameon=False, ncol=2, loc="upper right")
        elif sv_all is not None and not sv_all.empty:
            counts = sv_all.groupby("sample").size().reindex(samples, fill_value=0)
            ax_top.bar(x, counts.to_numpy(), width=0.82, color="#555555")
            ax_top.set_ylabel("SV calls", fontsize=_fs(8))
        else:
            burden = alt.groupby("sample")["altered"].sum().reindex(samples, fill_value=0)
            ax_top.bar(x, burden.to_numpy(), width=0.82, color="#555555")
            ax_top.set_ylabel("# genes\naltered", fontsize=_fs(8))
        ax_top.set_xlim(-0.5, n_cols - 0.5)
        ax_top.spines[["top", "right"]].set_visible(False)
        ax_top.tick_params(axis="y", labelsize=_fs(7))
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(axis="x", length=0)

        # ── main grid ───────────────────────────────────────────────────────
        ax_main.set_xlim(-0.5, n_cols - 0.5)
        ax_main.set_ylim(-0.5, n_rows - 0.5)
        ax_main.invert_yaxis()

        for r, gene in enumerate(genes):
            for c, sample in enumerate(samples):
                rec = idx.get((gene, sample))
                # background
                ax_main.add_patch(mpatches.Rectangle(
                    (c - 0.46, r - 0.42), 0.92, 0.84,
                    facecolor=ALT_COLORS["ABSENT"], linewidth=0))
                if rec is None:
                    continue

                # copy number — full-height block
                if rec["AMP"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r - 0.42), 0.92, 0.84,
                        facecolor=ALT_COLORS["AMP"], linewidth=0))
                elif rec["LOSS"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r - 0.42), 0.92, 0.84,
                        facecolor=ALT_COLORS["LOSS"], linewidth=0))

                # SV — a narrow vertical stripe, coloured by dominant type.
                # A wide horizontal bar covered most of the CNV block behind it
                # and read as a third background colour; a thin vertical mark
                # sits *over* the copy-number state instead of hiding it.
                if rec["SV"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.09, r - 0.42), 0.18, 0.84,
                        facecolor=SVTYPE_COLORS.get(rec["sv_type"], "#333333"),
                        linewidth=0, zorder=3))

                # SNV / indel — thin inner bar(s)
                if not show_snv:
                    pass
                elif rec["SNV"] and rec["INDEL"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r - 0.15), 0.92, 0.14,
                        facecolor=ALT_COLORS["SNV"], linewidth=0))
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r + 0.01), 0.92, 0.14,
                        facecolor=ALT_COLORS["INDEL"], linewidth=0))
                elif rec["SNV"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r - 0.10), 0.92, 0.20,
                        facecolor=ALT_COLORS["SNV"], linewidth=0))
                elif rec["INDEL"]:
                    ax_main.add_patch(mpatches.Rectangle(
                        (c - 0.46, r - 0.10), 0.92, 0.20,
                        facecolor=ALT_COLORS["INDEL"], linewidth=0))

        ax_main.set_yticks(range(n_rows))
        ax_main.set_yticklabels(genes, fontsize=_fs(9), style="italic")
        # `sample_labels` swaps the tick text without touching anything else —
        # used to emit a second copy of the figure carrying raw sequencing ids
        def _tick(s):
            return (sample_labels.get(s, s) if sample_labels
                    else SNVmethods._sample_label(s))
        _ha = "right" if label_rotation not in (0, 90) else "center"

        ax_main.set_xticks(range(n_cols))
        ax_main.set_xticklabels([_tick(s) for s in samples],
                                rotation=label_rotation,
                                fontsize=_fs(6) * xtick_scale, ha=_ha)
        if n_ann:
            plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(length=0)
        ax_main.spines[:].set_visible(False)

        # ── right: % of cohort altered per gene ─────────────────────────────
        freq = (alt.groupby("gene")["altered"].mean() * 100).reindex(genes).fillna(0)
        ax_right.barh(np.arange(n_rows), freq.to_numpy(), height=0.45,
                      color="#666666")
        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_xlabel("% cohort", fontsize=_fs(8))
        ax_right.tick_params(axis="x", labelsize=_fs(7))
        ax_right.set_yticks([])
        ax_right.spines[["top", "right", "left"]].set_visible(False)
        # Long bars get their label inside, short ones outside. With a narrow
        # panel an always-outside label runs into the legend.
        vmax = max(float(freq.max()), 1.0)
        ax_right.set_xlim(0, vmax * 1.08)
        for i, v in enumerate(freq.to_numpy()):
            inside = v > vmax * 0.55
            ax_right.text(v - vmax * 0.02 if inside else v + vmax * 0.03, i,
                          f"{v:.0f}%", va="center",
                          ha="right" if inside else "left",
                          color="white" if inside else "#333333",
                          fontsize=_fs(6.5))

        # ── annotation strips ───────────────────────────────────────────────
        ann_handles = []
        if n_ann:
            ax_ann.set_xlim(-0.5, n_cols - 0.5)
            ax_ann.set_ylim(-0.5, n_ann - 0.5)
            ax_ann.invert_yaxis()
            palette = plt.get_cmap("tab10")
            unknown_col = "#c9c9c9"
            any_unknown = False
            for i, name in enumerate(ann_names):
                mapping = annotations[name]
                present = {str(v) for s, v in mapping.items()
                           if s in samples and v is not None and str(v) != "nan"}
                if annotation_order and name in annotation_order:
                    levels = [str(lv) for lv in annotation_order[name]
                              if str(lv) in present]
                    levels += sorted(present - set(levels))
                    numeric = False
                else:
                    numeric = bool(present) and all(
                        re.fullmatch(r"-?\d+(\.\d+)?", lv) for lv in present)
                    levels = (sorted(present, key=float) if numeric
                              else sorted(present))
                # More levels than a palette can distinguish (a patient strip on
                # 40 patients) — colour is meaningless and the legend becomes a
                # wall. Band adjacent runs in two shades instead: what the strip
                # is actually for is showing where one patient's block ends.
                banded = len(levels) > 12
                if banded:
                    colors = {}
                elif numeric and len(levels) > 1:
                    # counts read better light->dark than as unrelated hues
                    seq = plt.get_cmap("YlOrRd")
                    colors = {lv: seq(0.25 + 0.65 * j / (len(levels) - 1))
                              for j, lv in enumerate(levels)}
                else:
                    colors = {lv: palette(j % 10) for j, lv in enumerate(levels)}

                # In banded mode the row's only job is to show which columns
                # belong to the same entity. Alternating on every column (as
                # happens when nearly every patient has one sample) is pure
                # noise, so shade only the runs longer than one and leave
                # singletons pale — the multi-sample patients then stand out.
                run_len = {}
                if banded:
                    keys = [str(mapping.get(s)) if mapping.get(s) is not None
                            else "nan" for s in samples]
                    i0 = 0
                    while i0 < len(keys):
                        i1 = i0
                        while i1 + 1 < len(keys) and keys[i1 + 1] == keys[i0]:
                            i1 += 1
                        run_len[i0] = i1 - i0 + 1
                        for j in range(i0, i1 + 1):
                            run_len[j] = i1 - i0 + 1
                        i0 = i1 + 1

                band, prev = 0, object()
                for c, sample in enumerate(samples):
                    v = mapping.get(sample)
                    key = str(v) if v is not None else "nan"
                    if banded:
                        if key != prev:
                            band ^= 1
                            prev = key
                        if key == "nan":
                            col = unknown_col
                            any_unknown = True
                        elif run_len.get(c, 1) > 1:
                            col = "#5b7fa6" if band else "#93b3d1"
                        else:
                            col = "#eef2f6"
                    else:
                        col = colors.get(key, unknown_col)
                        if key not in colors:
                            any_unknown = True
                    ax_ann.add_patch(mpatches.Rectangle(
                        (c - 0.46, i - 0.4), 0.92, 0.8,
                        facecolor=col, edgecolor="white", linewidth=0.3))
                if not banded:
                    ann_handles += [mpatches.Patch(facecolor=colors[lv],
                                                   label=f"{name}: {lv}")
                                    for lv in levels]
            # The "no metadata" key is deliberately not added. Samples with no
            # annotation are still drawn in unknown_col; the legend entry only
            # took up a line to restate what an empty-looking cell already says.
            ax_ann.set_yticks(range(n_ann))
            ax_ann.set_yticklabels(ann_names, fontsize=_fs(8))
            ax_ann.set_xticks(range(n_cols))
            ax_ann.set_xticklabels([_tick(s) for s in samples],
                                   rotation=label_rotation,
                                   fontsize=_fs(6) * xtick_scale, ha=_ha)
            ax_ann.tick_params(length=0)
            ax_ann.spines[:].set_visible(False)

        # ── legend ──────────────────────────────────────────────────────────
        handles = ([
            mpatches.Patch(facecolor=ALT_COLORS["SNV"],   label="SNV"),
            mpatches.Patch(facecolor=ALT_COLORS["INDEL"], label="indel"),
        ] if show_snv else []) + [
            mpatches.Patch(facecolor=ALT_COLORS["AMP"],   label="copy gain"),
            mpatches.Patch(facecolor=ALT_COLORS["LOSS"],  label="copy loss"),
        ]
        # SV is drawn as a narrow vertical stripe in the grid, so show it that
        # way in the legend too — a filled square reads as a background state,
        # which is what the CNV entries mean.
        # Read the types from `alt`, not from sv_all: alt is what the grid was
        # built from, so an SV class excluded by sv_exclude_types is absent here
        # and must not appear in the legend either.
        sv_seen = set()
        if "sv_type" in alt.columns:
            sv_seen = set(alt.loc[alt["SV"].astype(bool), "sv_type"].dropna().unique())
        elif sv_all is not None and not sv_all.empty and "svtype" in sv_all.columns:
            sv_seen = set(sv_all["svtype"].dropna().unique())
        handles += [mlines.Line2D([], [], color=col, lw=3.2, ls="-",
                                  marker="|", markersize=11, markeredgewidth=3.2,
                                  linestyle="none", label=f"SV: {svt}")
                    for svt, col in SVTYPE_COLORS.items()
                    if svt != "Total SV" and (not sv_seen or svt in sv_seen)]
        handles += [mpatches.Patch(facecolor=ALT_COLORS["ABSENT"], label="not altered")]
        handles += ann_handles
        ax_leg.legend(handles=handles,
                      fontsize=(legend_fontsize if legend_fontsize is not None
                                else ONCOPRINT_LEGEND_PT), loc="upper left",
                      bbox_to_anchor=(0.0, 1.45), frameon=False,
                      borderaxespad=0, labelspacing=0.35,
                      handlelength=1.4, handletextpad=0.5)

        fig.suptitle(f"{title}  (n={n_cols} samples, {n_rows} genes)",
                     fontsize=_fs(12), y=1.01)

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or f"cohort_oncoprint_{sort}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        return samples

    # ── cohort burden ────────────────────────────────────────────────────────

    @staticmethod
    def plot_cohort_burden(
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        samples: "list | None" = None,
        sort_by: str = "patient",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        Per-sample mutation burden across the cohort:
          top    — SNV / indel counts (stacked)
          bottom — SV counts stacked by SVTYPE

        sort_by : "patient" (default — cohort then patient number), "snv", "sv"
                  or "name". The value-sorted versions rank outliers, the
                  patient-sorted one is comparable with the other figures, so
                  save both; `filename` defaults to cohort_burden_<sort_by>.pdf
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if samples is None:
            pool = []
            if snv_all is not None and not snv_all.empty:
                pool += list(snv_all["sample"].unique())
            if sv_all is not None and not sv_all.empty:
                pool += list(sv_all["sample"].unique())
            samples = sorted(set(pool))
        if not samples:
            print("[WARN] No samples to plot.")
            return

        snv_counts = (snv_all.groupby(["sample", "vtype"]).size().unstack(fill_value=0)
                      if snv_all is not None and not snv_all.empty
                      else pd.DataFrame(index=samples))
        snv_counts = snv_counts.reindex(samples, fill_value=0)
        sv_counts  = (sv_all.groupby(["sample", "svtype"]).size().unstack(fill_value=0)
                      if sv_all is not None and not sv_all.empty
                      else pd.DataFrame(index=samples))
        sv_counts  = sv_counts.reindex(samples, fill_value=0)

        if sort_by == "patient":
            samples = CohortMethods.sort_samples(samples)
        elif sort_by == "snv" and not snv_counts.empty:
            samples = list(snv_counts.sum(axis=1).sort_values(ascending=False).index)
        elif sort_by == "sv" and not sv_counts.empty:
            samples = list(sv_counts.sum(axis=1).sort_values(ascending=False).index)
        snv_counts = snv_counts.reindex(samples, fill_value=0)
        sv_counts  = sv_counts.reindex(samples, fill_value=0)

        n = len(samples)
        x = np.arange(n)
        fig, axes = plt.subplots(2, 1, figsize=(max(9, n * 0.32), 7), sharex=True)

        bottom = np.zeros(n)
        for vt, col in (("SNV", ALT_COLORS["SNV"]), ("INDEL", ALT_COLORS["INDEL"])):
            vals = snv_counts[vt].to_numpy() if vt in snv_counts.columns else np.zeros(n)
            axes[0].bar(x, vals, bottom=bottom, width=0.82, color=col, label=vt)
            bottom += vals
        axes[0].set_ylabel("ClairS-TO PASS calls")
        axes[0].set_title("Small-variant burden per sample")
        axes[0].legend(fontsize=_fs(8), frameon=False)
        axes[0].spines[["top", "right"]].set_visible(False)
        if bottom.sum() > 0:
            axes[0].axhline(np.median(bottom), color="grey", ls="--", lw=0.8)
            axes[0].text(n - 0.5, np.median(bottom), f" median {np.median(bottom):,.0f}",
                         fontsize=_fs(7), va="bottom", ha="right", color="grey")

        bottom = np.zeros(n)
        for svt in [t for t in SVTYPES if t in sv_counts.columns]:
            vals = sv_counts[svt].to_numpy()
            axes[1].bar(x, vals, bottom=bottom, width=0.82,
                        color=SVTYPE_COLORS.get(svt, "#333333"), label=svt)
            bottom += vals
        axes[1].set_ylabel("SV calls")
        axes[1].set_title("Structural-variant burden per sample")
        axes[1].legend(fontsize=_fs(8), frameon=False, ncol=5)
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([SNVmethods._sample_label(s) for s in samples],
                                rotation=90, fontsize=_fs(7))
        axes[1].set_xlim(-0.5, n - 0.5)

        plt.tight_layout()
        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or f"cohort_burden_{sort_by}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

    # ── genome-wide cohort tracks ────────────────────────────────────────────

    @staticmethod
    def _genome_offsets(chroms: "list | None" = None) -> tuple[dict, list, int]:
        """Cumulative x-offset per chromosome for a linear whole-genome axis."""
        chroms = chroms or [c for c in CHROMS if c in HG38_CHROM_SIZES]
        offsets, ticks, running = {}, [], 0
        for c in chroms:
            offsets[c] = running
            ticks.append(running + HG38_CHROM_SIZES[c] / 2)
            running += HG38_CHROM_SIZES[c]
        return offsets, ticks, running

    @staticmethod
    def _draw_genome_axis(ax, chroms, offsets, total):
        for i, c in enumerate(chroms):
            if i % 2 == 0:
                ax.axvspan(offsets[c], offsets[c] + HG38_CHROM_SIZES[c],
                           color="#f2f2f2", zorder=0)
            ax.axvline(offsets[c], color="#cccccc", lw=0.4, zorder=1)
        ax.set_xlim(0, total)

    @staticmethod
    def _cnv_thresholds(cnv: pd.DataFrame, cnv_col: str, cnv_gain: float,
                        cnv_loss: float, cnv_baseline: str,
                        verbose: bool = True) -> tuple:
        """
        Per-sample (gain_thr, loss_thr) dicts.

        "absolute" — the same cutoffs for everyone. Right for CNVkit, whose cn
                     comes from a log2 ratio against a flat diploid reference.
        "ploidy"   — cutoffs become multiples of each sample's own baseline
                     (cnv_gain/2 and cnv_loss/2 times it), the baseline being
                     Wakhan's stated ploidy when present and the length-weighted
                     mean CN otherwise. These tumours run from ploidy 1.63 to
                     3.61, so a fixed "cn ≥ 2.5 = gain" calls most of a
                     ploidy-3.6 genome amplified and most of a low-ploidy genome
                     deleted.

        Shared by build_cohort_alterations and plot_cnv_frequency_genome so the
        oncoprint and the frequency track cannot disagree about the same
        segments.
        """
        gain_thr, loss_thr = {}, {}
        if cnv_baseline == "ploidy":
            ref = cnv.assign(_len=(cnv["end"] - cnv["start"]).clip(lower=1))
            bases = {}
            for s, g in ref.groupby("sample"):
                if "ploidy" in g.columns and g["ploidy"].notna().any():
                    base = float(g["ploidy"].dropna().iloc[0])
                else:
                    base = float(np.average(g[cnv_col], weights=g["_len"]))
                base = base if base > 0 else 2.0
                bases[s] = base
                gain_thr[s] = base * (cnv_gain / 2.0)
                loss_thr[s] = base * (cnv_loss / 2.0)
            if verbose and bases:
                print(f"[INFO] CNV thresholds relative to each sample's baseline "
                      f"({cnv_gain / 2.0:.2f}x gain / {cnv_loss / 2.0:.2f}x loss); "
                      f"baselines {min(bases.values()):.2f}–{max(bases.values()):.2f}")
        else:
            for s in cnv["sample"].unique():
                gain_thr[s], loss_thr[s] = cnv_gain, cnv_loss
        return gain_thr, loss_thr

    @staticmethod
    def plot_cnv_frequency_genome(
        cnv_all: pd.DataFrame,
        samples: "list | None" = None,
        bin_size: int = 1_000_000,
        gain_cn: float = 2.5,
        loss_cn: float = 1.5,
        cnv_baseline: str = "absolute",
        mask_centromeres: bool = True,
        telomere_bp: int = 2_000_000,
        centromere_pad: int = 2_000_000,
        flag_pct: float = 60.0,
        include_sex: bool = False,
        cnv_col: str = "cn",
        mark_genes: "list | None" = None,
        title: str = "Cohort copy-number frequency",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        GISTIC-style cohort CNV frequency: for every 1 Mb bin, the percentage of
        samples with a gain (up, red) or a loss (down, blue).

        Works with any segment table carrying chrom/start/end/<cnv_col>/sample.
        With CORAL amplicon segments this reads as a recurrent-amplification
        track (CORAL only reports amplified regions, so the loss side will be
        empty by construction).

        `cnv_baseline` must match the source, and must match whatever the
        oncoprint used for the same segments — see `_cnv_thresholds`. Wakhan is
        purity/ploidy-corrected, so it needs "ploidy"; CNVkit is a ratio against
        a flat diploid reference, so it needs "absolute".

        A bin can carry both a gain and a loss bar. Usually that is two
        different samples. Within one sample it means two overlapping segments,
        which for haplotype-resolved Wakhan calls is a real event: one haplotype
        amplified, the other lost.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if cnv_all is None or cnv_all.empty:
            print("[INFO] No CNV segments available — skipping cohort CNV frequency plot.")
            return
        need = {"chrom", "start", "end", cnv_col, "sample"}
        if not need.issubset(cnv_all.columns):
            print(f"[WARN] CNV table missing columns {need - set(cnv_all.columns)}")
            return

        samples = samples or sorted(cnv_all["sample"].unique())
        n_samples = len(samples)
        chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
        if not include_sex:
            # X and Y frequencies are dominated by sex and by ploidy
            # normalisation, not by tumour biology: at median ploidy 3.3 a
            # normal two-copy X scores as a loss in both sexes, and Y is absent
            # in every female. Plotting them next to autosomes invites a
            # reading they cannot support.
            chroms = [c for c in chroms if c not in ("X", "Y")]
        offsets, ticks, total = CohortMethods._genome_offsets(chroms)

        n_bins = {c: int(np.ceil(HG38_CHROM_SIZES[c] / bin_size)) for c in chroms}
        gain = {c: np.zeros(n_bins[c]) for c in chroms}
        loss = {c: np.zeros(n_bins[c]) for c in chroms}

        # Bins to blank. Centromeres and the first/last few Mb are unmappable
        # or repeat-rich: segment callers emit both a gain and a loss over the
        # same bin there, which produced 91% gain AND 98% loss at chr16:46 Mb
        # and a 100% loss bar at the start of every chromosome. Those are the
        # tallest bars in an unmasked track and they are all artifact.
        masked = {c: np.zeros(n_bins[c], dtype=bool) for c in chroms}
        for c in chroms:
            size = HG38_CHROM_SIZES[c]
            if telomere_bp:
                masked[c][:max(1, int(telomere_bp // bin_size))] = True
                masked[c][-max(1, int(telomere_bp // bin_size)):] = True
            if mask_centromeres:
                spans = []
                if c in HG38_CENTROMERES:
                    a, b = HG38_CENTROMERES[c]
                    # pad: the acen band alone is narrower than the region the
                    # caller misbehaves over
                    spans.append((a - centromere_pad, b + centromere_pad))
                spans += HG38_HETEROCHROMATIN.get(c, [])
                for a, b in spans:
                    masked[c][max(0, int(a // bin_size)):
                              int(b // bin_size) + 1] = True
        n_masked = sum(int(m.sum()) for m in masked.values())
        if n_masked:
            print(f"[INFO] {n_masked} bin(s) blanked as centromeric/telomeric "
                  f"(mask_centromeres/telomere_bp turn this off).")

        df = cnv_all.assign(_chr=cnv_all["chrom"].map(CohortMethods._norm_chrom))
        gain_thr, loss_thr = CohortMethods._cnv_thresholds(
            df, cnv_col, gain_cn, loss_cn, cnv_baseline)

        for sample in samples:
            sub = df.loc[df["sample"] == sample]
            g_cut = gain_thr.get(sample, gain_cn)
            l_cut = loss_thr.get(sample, loss_cn)
            seen_gain = {c: np.zeros(n_bins[c], dtype=bool) for c in chroms}
            seen_loss = {c: np.zeros(n_bins[c], dtype=bool) for c in chroms}
            for _, row in sub.iterrows():
                c = row["_chr"]
                if c not in n_bins:
                    continue
                b0 = int(max(row["start"], 0) // bin_size)
                b1 = int(min(row["end"], HG38_CHROM_SIZES[c]) // bin_size)
                b1 = min(b1, n_bins[c] - 1)
                if b1 < b0:
                    continue
                if row[cnv_col] >= g_cut:
                    seen_gain[c][b0:b1 + 1] = True
                elif row[cnv_col] <= l_cut:
                    seen_loss[c][b0:b1 + 1] = True
            for c in chroms:
                gain[c] += seen_gain[c]
                loss[c] += seen_loss[c]

        fig, ax = plt.subplots(figsize=(15, 4.5))
        CohortMethods._draw_genome_axis(ax, chroms, offsets, total)
        for c in chroms:
            xs = offsets[c] + np.arange(n_bins[c]) * bin_size
            g_pct = np.where(masked[c], 0.0, gain[c] / n_samples * 100)
            l_pct = np.where(masked[c], 0.0, loss[c] / n_samples * 100)
            ax.bar(xs, g_pct, width=bin_size, color="#c1272d",
                   align="edge", linewidth=0)
            ax.bar(xs, -l_pct, width=bin_size, color="#2c7fb8",
                   align="edge", linewidth=0)
            # show where the blanks are, so a gap is not read as "no events"
            for i0 in np.flatnonzero(masked[c]):
                ax.axvspan(offsets[c] + i0 * bin_size,
                           offsets[c] + (i0 + 1) * bin_size,
                           color="#eeeeee", zorder=0, linewidth=0)

        # Whatever the mask misses, say so rather than let a 90% bar stand as
        # if it were biology. A single bin at near-100% in a cohort this size is
        # almost always unmappable sequence, not a shared event.
        flagged = []
        for c in chroms:
            for i0 in np.flatnonzero(~masked[c]):
                for arr, kind in ((gain[c], "gain"), (loss[c], "loss")):
                    pct = arr[i0] / n_samples * 100
                    if pct >= flag_pct:
                        flagged.append((c, i0 * bin_size, kind, pct))
        if flagged:
            print(f"[WARN] {len(flagged)} unmasked bin(s) at >={flag_pct:.0f}% — "
                  f"check these are not unmappable regions:")
            for c, pos, kind, pct in flagged[:8]:
                print(f"         chr{c}:{pos/1e6:.0f}-{pos/1e6 + bin_size/1e6:.0f} Mb "
                      f"{kind} {pct:.0f}%")
            if len(flagged) > 8:
                print(f"         ... and {len(flagged) - 8} more")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(chroms, fontsize=_fs(8))
        ax.set_ylabel("% of cohort")
        if cnv_baseline == "ploidy":
            thr_txt = (f"gain ≥{gain_cn / 2.0:.2g}x / loss ≤{loss_cn / 2.0:.2g}x "
                       f"each sample's ploidy")
        else:
            thr_txt = f"gain CN≥{gain_cn} / loss CN≤{loss_cn}"
        ax.set_title(f"{title}  (n={n_samples}, {bin_size//1000} kb bins, {thr_txt})")
        ax.spines[["top", "right"]].set_visible(False)
        ymax = max(1.0, max(float(np.where(masked[c], 0, gain[c]).max())
                            for c in chroms) / n_samples * 100)
        ymin = -max(1.0, max(float(np.where(masked[c], 0, loss[c]).max())
                             for c in chroms) / n_samples * 100)
        ax.set_ylim(ymin * 1.25 - 1, ymax * 1.25 + 1)

        for g in (mark_genes or []):
            try:
                name, coords = CohortMethods.resolve_gene(g)
            except KeyError:
                continue
            c = CohortMethods._norm_chrom(coords["chrom"])
            if c not in offsets:
                continue
            xpos = offsets[c] + (coords["start"] + coords["end"]) / 2
            ax.axvline(xpos, color="black", lw=0.6, ls=":", zorder=3)
            ax.text(xpos, ax.get_ylim()[1], f" {name}", rotation=90,
                    fontsize=_fs(6.5), va="top", ha="left")

        plt.tight_layout()
        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "cohort_cnv_frequency.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_sv_breakpoint_recurrence(
        sv_all: pd.DataFrame,
        samples: "list | None" = None,
        bin_size: int = 1_000_000,
        svtypes: "list | None" = None,
        min_sv_len: int = 10_000,
        mark_genes: "list | None" = None,
        title: str = "Recurrent SV breakpoints across the cohort",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        Fraction of the cohort carrying at least one SV breakpoint in each 1 Mb
        bin. Both ends of every call are counted (POS and END/CHR2), so
        translocation partners show up on both chromosomes.

        `min_sv_len` (default 10 kb, BNDs exempt) matters as much here as in the
        oncoprint: a long-read callset carries hundreds of thousands of small
        INS/DEL, which put a breakpoint in essentially every bin of every sample
        and flatten the track at 100%.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)

        if sv_all is None or sv_all.empty:
            print("[INFO] No SV table available — skipping breakpoint recurrence plot.")
            return

        df = sv_all
        if svtypes:
            df = df.loc[df["svtype"].isin(svtypes)]
        if min_sv_len and "svlen" in df.columns:
            before = len(df)
            df = df.loc[(df["svtype"] == "BND") | (df["svlen"].abs() >= min_sv_len)]
            print(f"[INFO] SV size filter ≥{min_sv_len:,} bp (BND exempt): "
                  f"{before:,} → {len(df):,} calls")
        if df.empty:
            print("[INFO] No SVs left after filtering — skipping recurrence plot.")
            return
        samples = samples or sorted(df["sample"].unique())
        n_samples = len(samples)
        chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
        offsets, ticks, total = CohortMethods._genome_offsets(chroms)
        n_bins = {c: int(np.ceil(HG38_CHROM_SIZES[c] / bin_size)) for c in chroms}

        recur = {c: np.zeros(n_bins[c]) for c in chroms}
        for sample in samples:
            sub = df.loc[df["sample"] == sample]
            seen = {c: np.zeros(n_bins[c], dtype=bool) for c in chroms}
            for chrom_col, pos_col in (("chrom", "start"), ("chrom2", "end")):
                if chrom_col not in sub.columns:
                    continue
                cc = sub[chrom_col].map(CohortMethods._norm_chrom)
                for c, p in zip(cc, sub[pos_col]):
                    if c not in n_bins:
                        continue
                    b = int(min(max(p, 0), HG38_CHROM_SIZES[c] - 1) // bin_size)
                    seen[c][min(b, n_bins[c] - 1)] = True
            for c in chroms:
                recur[c] += seen[c]

        fig, ax = plt.subplots(figsize=(15, 4))
        CohortMethods._draw_genome_axis(ax, chroms, offsets, total)
        for c in chroms:
            xs = offsets[c] + np.arange(n_bins[c]) * bin_size
            ax.bar(xs, recur[c] / n_samples * 100, width=bin_size,
                   color="#6a51a3", align="edge", linewidth=0)

        ax.set_xticks(ticks)
        ax.set_xticklabels(chroms, fontsize=_fs(8))
        ax.set_ylabel("% of cohort with a breakpoint")
        svt_note = f" [{', '.join(svtypes)}]" if svtypes else ""
        ax.set_title(f"{title}{svt_note}  (n={n_samples}, {bin_size//1000} kb bins)")
        ax.spines[["top", "right"]].set_visible(False)

        for g in (mark_genes or []):
            try:
                name, coords = CohortMethods.resolve_gene(g)
            except KeyError:
                continue
            c = CohortMethods._norm_chrom(coords["chrom"])
            if c not in offsets:
                continue
            xpos = offsets[c] + (coords["start"] + coords["end"]) / 2
            ax.axvline(xpos, color="black", lw=0.6, ls=":", zorder=3)
            ax.text(xpos, ax.get_ylim()[1], f" {name}", rotation=90,
                    fontsize=_fs(6.5), va="top", ha="left")

        plt.tight_layout()
        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "cohort_sv_recurrence.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_cohort_chrom_heatmap(
        df: pd.DataFrame,
        samples: "list | None" = None,
        value: str = "count",
        log_scale: bool = True,
        cmap: str = "viridis",
        title: str = "Events per chromosome",
        sort: str = "patient",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        Sample x chromosome event-count heatmap. Works with either the SNV or
        the SV cohort table (anything with `sample` and `chrom`).

        sort     : "patient" (default) or "total" (busiest sample first)
        filename : required when calling this more than once into the same
                   save_dir — otherwise the SNV and SV heatmaps overwrite each
                   other. Defaults to cohort_chrom_heatmap_<sort>.pdf
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if df is None or df.empty:
            print("[INFO] Empty table — skipping chromosome heatmap.")
            return

        chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
        d = df.assign(_chr=df["chrom"].map(CohortMethods._norm_chrom))
        mat = (d.groupby(["sample", "_chr"]).size().unstack(fill_value=0)
                 .reindex(columns=chroms, fill_value=0))
        if samples:
            mat = mat.reindex(samples, fill_value=0)
        if sort == "patient":
            mat = mat.reindex(CohortMethods.sort_samples(mat.index))
        else:
            mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]

        vals = np.log10(mat.to_numpy() + 1) if log_scale else mat.to_numpy()
        fig, ax = plt.subplots(figsize=(10, max(4, len(mat) * 0.22)))
        im = ax.imshow(vals, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(chroms)))
        ax.set_xticklabels(chroms, fontsize=_fs(8))
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels([SNVmethods._sample_label(s) for s in mat.index], fontsize=_fs(7))
        ax.set_xlabel("chromosome")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("log10(count + 1)" if log_scale else "count", fontsize=_fs(8))

        plt.tight_layout()
        if actual_dir is not None:
            fpath = os.path.join(actual_dir,
                                 filename or f"cohort_chrom_heatmap_{sort}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return mat

    # ── methylation: cohort subtyping ────────────────────────────────────────

    @staticmethod
    def _find_bedmethyl(modkit_root: str, sample: str) -> "str | None":
        """
        Locate a modkit pileup bedMethyl for `sample`.

        Covers both layouts in use: flat ({root}/{sample}_CpG_5mC.bed[.gz]) and
        the per-sample tree the LSF pipeline writes,
        {root}/{sample}/3_methylation/{sample}_CpG_5mC.bed[.gz].
        `.bed.gz.tbi` indexes never match, since every glob ends at .bed/.bed.gz.
        """
        patterns = [
            # what modkit_scripts/step3_pileup.lsf writes — per-sample tree first
            os.path.join(modkit_root, sample, "3_methylation", f"{sample}_CpG_5mC.bed"),
            os.path.join(modkit_root, sample, "3_methylation", f"{sample}_CpG_5mC.bed.gz"),
            os.path.join(modkit_root, sample, "3_methylation", "*.bed"),
            os.path.join(modkit_root, sample, "3_methylation", "*.bed.gz"),
            os.path.join(modkit_root, f"{sample}_CpG_5mC.bed"),
            os.path.join(modkit_root, f"{sample}_CpG_5mC.bed.gz"),
            os.path.join(modkit_root, sample, "*.bed"),
            os.path.join(modkit_root, sample, "*.bed.gz"),
            os.path.join(modkit_root, sample, "*", "*.bed"),
            os.path.join(modkit_root, sample, "*", "*.bed.gz"),
            os.path.join(modkit_root, f"{sample}*.bed"),
            os.path.join(modkit_root, f"{sample}*.bed.gz"),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def load_regions_bed(path: str) -> pd.DataFrame:
        """
        Load a plain BED of regions (CpG islands, promoters, ...) as
        chrom/start/end/name. Handles UCSC `cpgIslandExt.txt.gz`, which carries a
        leading bin column.
        """
        opener = gzip.open if str(path).endswith(".gz") else open
        rows = []
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith(("#", "track", "browser")):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 3:
                    continue
                # UCSC cpgIslandExt: bin, chrom, start, end, name, ...
                if f[0].isdigit() and len(f) > 4 and f[1].startswith("chr"):
                    f = f[1:]
                chrom = CohortMethods._norm_chrom(f[0])
                if chrom not in PRIMARY_CHROMS:
                    continue
                try:
                    start, end = int(f[1]), int(f[2])
                except ValueError:
                    continue
                name = f[3] if len(f) > 3 else f"{chrom}:{start}-{end}"
                rows.append((chrom, start, end, name))
        df = pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])

        # `name` is NOT an identifier: UCSC cpgIslandExt writes the CpG count
        # ("CpG: 111"), so 32,038 islands carry only ~485 distinct names. Keying
        # anything by it silently collapses the region set. region_id is the
        # locus itself and is what the beta matrix is indexed by.
        df = CohortMethods._ensure_region_id(df)
        n_raw = len(df)
        df = df.drop_duplicates(subset="region_id").reset_index(drop=True)
        dropped = n_raw - len(df)
        print(f"Loaded {len(df):,} regions from {os.path.basename(path)}"
              + (f" ({dropped:,} duplicate loci dropped)" if dropped else ""))
        return df

    @staticmethod
    def _ensure_region_id(regions: pd.DataFrame) -> pd.DataFrame:
        """Add a unique per-locus `region_id` (chrom:start-end) if absent."""
        if "region_id" in regions.columns:
            return regions
        return regions.assign(
            region_id=(regions["chrom"].astype(str) + ":"
                       + regions["start"].astype(str) + "-"
                       + regions["end"].astype(str)))

    @staticmethod
    def _beta_for_sample(bed_path: str, sample: str, regions: pd.DataFrame,
                         min_cov: int) -> "pd.Series | None":
        """
        Mean methylation fraction per region for one sample.

        modkit bedMethyl columns (0-based): 0 chrom, 1 start, 2 end, 3 mod code,
        4 score, 5 strand, 6-8 display, 9 valid_coverage, 10 percent_modified.
        CpGs below `min_cov` valid reads are dropped before averaging.
        """
        opener = gzip.open if bed_path.endswith(".gz") else open
        recs = []
        try:
            with opener(bed_path, "rt") as fh:
                for line in fh:
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 11:
                        continue
                    if f[3] not in ("m", "C", "5mC"):      # 5mC records only
                        continue
                    try:
                        cov = int(f[9])
                        if cov < min_cov:
                            continue
                        recs.append((CohortMethods._norm_chrom(f[0]), int(f[1]), float(f[10])))
                    except ValueError:
                        continue
        except OSError as e:
            print(f"[WARN] Could not read {bed_path}: {e}")
            return None

        if not recs:
            print(f"[WARN] No CpGs passed min_cov={min_cov} for {sample}")
            return None

        cpg = pd.DataFrame(recs, columns=["chrom", "pos", "pct"])
        regions = CohortMethods._ensure_region_id(regions)
        out = {}
        for chrom, reg in regions.groupby("chrom"):
            sub = cpg.loc[cpg["chrom"] == chrom]
            if sub.empty:
                continue
            order = np.argsort(sub["pos"].to_numpy())
            pos   = sub["pos"].to_numpy()[order]
            pct   = sub["pct"].to_numpy()[order]
            csum  = np.concatenate([[0.0], np.cumsum(pct)])
            lo = np.searchsorted(pos, reg["start"].to_numpy(), side="left")
            hi = np.searchsorted(pos, reg["end"].to_numpy(),   side="right")
            n  = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
            # keyed by locus, not by `name` — see load_regions_bed
            for rid, m in zip(reg["region_id"].to_numpy(), means):
                out[rid] = m / 100.0         # percent -> beta value

        return pd.Series(out, name=sample, dtype="float64")

    @staticmethod
    def load_methylation_cohort(
        modkit_root: str,
        samples: list,
        regions: "pd.DataFrame | str",
        min_cov: int = 5,
        min_sample_frac: float = 0.75,
        n_jobs: int = 4,
        cache_path: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Build a region x sample beta-value matrix from modkit bedMethyl pileups
        — the input for cohort methylation subtyping.

        regions : DataFrame from load_regions_bed, or a path to a BED / the UCSC
                  cpgIslandExt file. CpG islands are the usual choice (this is
                  the same feature set toronto_clustering.py uses).

        Regions covered in fewer than `min_sample_frac` of samples are dropped.
        Returns an empty frame (with a warning) when the pileups aren't there yet.
        """
        if cache_path and os.path.exists(cache_path):
            beta = CohortMethods._cache_read(cache_path)
            # Caches written before regions were keyed by locus are indexed by
            # the UCSC name column ("CpG: 111"), which collapsed ~32k islands
            # into ~485 rows. Recompute rather than trust one.
            if beta is not None and not beta.empty and not all(
                    isinstance(i, str) and ":" in i and "-" in i
                    for i in beta.index[:20]):
                print(f"[INFO] Ignoring stale beta cache ({beta.shape[0]:,} rows, "
                      f"not locus-indexed): {cache_path}")
                beta = None
            if beta is not None and samples:
                extra = [c for c in beta.columns if c not in set(samples)]
                if extra:
                    print(f"[INFO] dropping {len(extra)} sample(s) from the cached "
                          f"beta matrix, not in the requested list: "
                          f"{', '.join(map(str, extra[:4]))}")
                    beta = beta[[c for c in beta.columns if c in set(samples)]]
            if beta is not None:
                print(f"Loaded cached beta matrix {beta.shape} from {cache_path}")
                return beta

        if isinstance(regions, str):
            regions = CohortMethods.load_regions_bed(regions)
        if regions.empty:
            print("[WARN] No regions supplied — cannot build a beta matrix.")
            return pd.DataFrame()

        found = {}
        for s in samples:
            path = CohortMethods._find_bedmethyl(modkit_root, s)
            if path:
                found[s] = path
        missing = [s for s in samples if s not in found]
        if missing:
            print(f"[WARN] No bedMethyl found for {len(missing)}/{len(samples)} samples "
                  f"under {modkit_root} (e.g. {missing[0]})")
        if not found:
            print("[INFO] Methylation pileups not available yet — "
                  "run modkit pileup for this cohort, then re-run this cell.")
            return pd.DataFrame()

        print(f"Reading {len(found)} bedMethyl files (n_jobs={n_jobs})...")
        series = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
            joblib.delayed(CohortMethods._beta_for_sample)(path, s, regions, min_cov)
            for s, path in found.items()
        )
        series = [s for s in series if s is not None]
        if not series:
            print("[WARN] No methylation data could be summarised.")
            return pd.DataFrame()

        beta = pd.concat(series, axis=1)
        keep = beta.notna().mean(axis=1) >= min_sample_frac
        beta = beta.loc[keep]
        print(f"Beta matrix: {beta.shape[0]:,} regions x {beta.shape[1]} samples "
              f"(covered in ≥{min_sample_frac:.0%} of samples)")

        if cache_path:
            CohortMethods._cache_write(beta, cache_path)
        return beta

    @staticmethod
    def load_methylation_multi(
        roots_by_cohort: dict,
        samples_by_cohort: dict,
        regions: "pd.DataFrame | str",
        sample_map: "dict | None" = None,
        min_cov: int = 5,
        min_sample_frac: float = 0.75,
        n_jobs: int = 4,
        cache_dir: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Build one beta matrix spanning several cohorts, using whichever cohorts
        actually have modkit pileups on disk.

            beta = CohortMethods.load_methylation_multi(
                roots_by_cohort   = {"Cohort1": METHYL_ROOT_C1,
                                     "Cohort2": METHYL_ROOT_C2},   # add later
                samples_by_cohort = {"Cohort1": SAMPLES_C1,
                                     "Cohort2": SAMPLES_C2},
                regions           = CPG_ISLANDS,
                sample_map        = SAMPLE_MAP,
            )

        A cohort whose root is missing or empty is reported and skipped, so the
        call is written once and starts including the second cohort the moment
        its pileups land — no other edit needed. Columns are renamed through
        `sample_map` (Cohort2-P1 …), and regions are intersected across cohorts
        so every sample is comparable.

        Note the batch caveat: methylation clustering of two separately
        basecalled//sequenced cohorts can separate by run as much as by biology.
        Colour the embedding by `cohort` before reading anything into it.
        """
        if isinstance(regions, str):
            regions = CohortMethods.load_regions_bed(regions)
        if regions is None or regions.empty:
            print("[WARN] No regions supplied — cannot build a beta matrix.")
            return pd.DataFrame()

        mats, used = [], []
        for cohort, root in roots_by_cohort.items():
            samples = samples_by_cohort.get(cohort, [])
            if not root:
                print(f"[INFO] {cohort}: no methylation root configured — "
                      f"skipped, set one once the pileups exist.")
                continue
            if not os.path.isdir(root):
                print(f"[INFO] {cohort}: methylation root not present "
                      f"({root}) — skipped, add it when the pileups are ready.")
                continue
            cache = (os.path.join(cache_dir, f"methylation_beta_{cohort}.pkl")
                     if cache_dir else None)
            print(f"\n─── {cohort} (run) ───")
            beta = CohortMethods.load_methylation_cohort(
                root, samples, regions,
                min_cov=min_cov, min_sample_frac=min_sample_frac,
                n_jobs=n_jobs, cache_path=cache,
            )
            if beta is None or beta.empty:
                continue
            if sample_map:
                beta = beta.rename(columns={c: sample_map.get(c, c)
                                            for c in beta.columns})
            mats.append(beta)
            used.append(f"{cohort} (n={beta.shape[1]})")

        if not mats:
            print("\n[INFO] No cohort has methylation pileups on disk yet — "
                  "the methylation plots below will skip.")
            return pd.DataFrame()

        combined = pd.concat(mats, axis=1, join="inner")
        print(f"\nCombined beta matrix: {combined.shape[0]:,} shared regions x "
              f"{combined.shape[1]} samples — {', '.join(used)}")
        if len(mats) > 1:
            print("[NOTE] Multiple cohorts combined — check the embedding for a "
                  "batch split before interpreting clusters biologically.")
        return combined

    @staticmethod
    def _top_variable(beta: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Rows with the highest across-sample variance, NaNs row-mean imputed."""
        b = beta.copy()
        b = b.apply(lambda r: r.fillna(r.mean()), axis=1)
        var = b.var(axis=1, skipna=True)
        return b.loc[var.sort_values(ascending=False).head(top_n).index]

    @staticmethod
    def plot_methylation_umap(
        beta: pd.DataFrame,
        top_n_var: int = 5000,
        method: str = "umap",
        n_clusters: int = 3,
        annotations: "dict | None" = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        title: str = "Chordoma — methylation embedding",
        save_dir: "str | None" = None,
    ) -> dict:
        """
        UMAP (or t-SNE) embedding of the cohort's methylation profiles, coloured
        by hierarchical-clustering label — the cohort counterpart of the CNS
        methylation-classifier plots.

        Returns {sample: cluster_label}; feed that straight into
        plot_cohort_oncoprint(annotations={"Methylation cluster": labels}).
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if beta is None or beta.empty:
            print("[INFO] No beta matrix — skipping methylation embedding.")
            return {}
        if beta.shape[1] < 3:
            print(f"[WARN] Only {beta.shape[1]} samples with methylation — "
                  "an embedding needs at least 3.")
            return {}

        from scipy.cluster.hierarchy import linkage, fcluster

        X = CohortMethods._top_variable(beta, top_n_var).T      # samples x regions
        samples = list(X.index)

        n_clusters = min(n_clusters, len(samples))
        Z = linkage(X.to_numpy(), method="ward", metric="euclidean")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        cluster_map = {s: f"MC{l}" for s, l in zip(samples, labels)}

        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(samples) - 1),
                                    min_dist=min_dist, random_state=random_state)
                emb = reducer.fit_transform(X.to_numpy())
                axis_label = "UMAP"
            except ImportError:
                print("[WARN] umap-learn not installed — falling back to t-SNE.")
                method = "tsne"
        if method != "umap":
            from sklearn.manifold import TSNE
            perplexity = max(2, min(30, len(samples) - 1))
            emb = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state, init="pca").fit_transform(X.to_numpy())
            axis_label = "t-SNE"

        n_panels = 1 + (len(annotations) if annotations else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
        axes = axes[0]

        palette = plt.get_cmap("tab10")
        levels = sorted(set(cluster_map.values()))
        colors = {lv: palette(i % 10) for i, lv in enumerate(levels)}
        for lv in levels:
            m = [i for i, s in enumerate(samples) if cluster_map[s] == lv]
            axes[0].scatter(emb[m, 0], emb[m, 1], s=70, color=colors[lv],
                            edgecolor="white", linewidth=0.6, label=f"{lv} (n={len(m)})")
        for i, s in enumerate(samples):
            axes[0].annotate(SNVmethods._sample_label(s), (emb[i, 0], emb[i, 1]),
                             fontsize=_fs(6), xytext=(4, 3), textcoords="offset points")
        axes[0].set_title(f"Methylation clusters (ward, k={n_clusters})")
        axes[0].legend(fontsize=_fs(8), frameon=False)

        for ax, name in zip(axes[1:], (annotations or {}).keys()):
            mapping = annotations[name]
            lv = sorted({str(v) for v in mapping.values() if v is not None})
            cols = {v: palette(i % 10) for i, v in enumerate(lv)}
            for v in lv:
                m = [i for i, s in enumerate(samples) if str(mapping.get(s)) == v]
                if m:
                    ax.scatter(emb[m, 0], emb[m, 1], s=70, color=cols[v],
                               edgecolor="white", linewidth=0.6, label=v)
            ax.set_title(name)
            ax.legend(fontsize=_fs(8), frameon=False)

        for ax in axes:
            ax.set_xlabel(f"{axis_label} 1")
            ax.set_ylabel(f"{axis_label} 2")
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"{title}  ({X.shape[1]:,} most variable regions, "
                     f"n={len(samples)} samples)", fontsize=_fs(12))
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, f"methylation_{axis_label.lower()}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        return cluster_map



    @staticmethod
    def plot_methylation_consensus(
        beta: pd.DataFrame,
        top_n_var: int = 5000,
        n_clusters: int = 3,
        n_iter: int = 1000,
        subsample_frac: float = 0.80,
        annotations: "dict | None" = None,
        random_state: int = 42,
        title: str = "Methylation consensus clustering",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> dict:
        """
        Consensus-clustering matrix: sample x sample, each cell the fraction of
        resampling runs in which that pair landed in the same cluster.

        Same construction as the Toronto/Sonopet projection work — subsample
        `subsample_frac` of the regions `n_iter` times, Ward-cluster each
        subsample, and count co-assignment. A single Ward run tells you what the
        partition IS; the consensus matrix tells you how much of that partition
        survives perturbing the feature set, which is the part that matters when
        the sample count is small.

        Read it as: solid blocks on the diagonal = a cluster that reproduces;
        washed-out blocks = a partition the data does not really support.

        Returns {"consensus": the matrix, "labels": {sample: "MC<k>"},
                 "order": leaf order}.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import squareform, pdist

        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if beta is None or beta.empty:
            print("[INFO] No beta matrix — skipping consensus clustering.")
            return {}
        n = beta.shape[1]
        if n < 3:
            print(f"[WARN] Only {n} sample(s) with methylation — a consensus "
                  f"matrix needs at least 3.")
            return {}

        X = CohortMethods._top_variable(beta, top_n_var).T     # samples x regions
        samples = list(X.index)
        k = min(n_clusters, n)
        rng = np.random.RandomState(random_state)

        co = np.zeros((n, n), dtype=float)
        sel = np.zeros((n, n), dtype=float)
        n_feat = X.shape[1]
        take = max(2, int(round(subsample_frac * n_feat)))
        for _ in range(n_iter):
            cols = rng.choice(n_feat, size=take, replace=False)
            sub = X.iloc[:, cols].to_numpy()
            try:
                Z_it = linkage(sub, method="ward")
                lab = fcluster(Z_it, t=k, criterion="maxclust")
            except (ValueError, RuntimeError):
                continue
            same = (lab[:, None] == lab[None, :]).astype(float)
            co += same
            sel += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            cons = np.where(sel > 0, co / np.maximum(sel, 1), 0.0)
        np.fill_diagonal(cons, 1.0)
        cons_df = pd.DataFrame(cons, index=samples, columns=samples)

        # final partition on the consensus matrix itself
        dist = np.clip(1.0 - cons, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
        label_map = {s: f"MC{l}" for s, l in zip(samples, labels)}
        order_idx = dendrogram(Z, no_plot=True)["leaves"]
        order = [samples[i] for i in order_idx]

        # how reproducible is each cluster? mean within-cluster consensus
        summary = []
        for lv in sorted(set(labels)):
            mem = [s for s, l in zip(samples, labels) if l == lv]
            if len(mem) < 2:
                summary.append((f"MC{lv}", len(mem), np.nan))
                continue
            sub = cons_df.loc[mem, mem].to_numpy()
            iu = np.triu_indices(len(mem), k=1)
            summary.append((f"MC{lv}", len(mem), float(sub[iu].mean())))
        summary = pd.DataFrame(summary, columns=["cluster", "n", "mean_consensus"])

        # ── figure ──────────────────────────────────────────────────────────
        ann_names = list(annotations.keys()) if annotations else []
        n_ann = len(ann_names)
        fig = plt.figure(figsize=(max(7.0, 0.42 * n + 4.0),
                                  max(7.0, 0.42 * n + 3.4)))
        gs = fig.add_gridspec(
            2 + n_ann + 1, 2,
            height_ratios=[1.4] + [0.16] * n_ann + [8.0, 0.001],
            width_ratios=[10, 0.45], hspace=0.04, wspace=0.03)

        ax_d = fig.add_subplot(gs[0, 0])
        dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0,
                   link_color_func=lambda _: "#666666")
        ax_d.axis("off")

        palette = plt.get_cmap("tab10")
        ann_handles = []
        for i, name in enumerate(ann_names):
            axa = fig.add_subplot(gs[1 + i, 0])
            mapping = annotations[name]
            levels = sorted({str(v) for v in mapping.values() if v is not None})
            cols = {lv: palette(j % 10) for j, lv in enumerate(levels)}
            for xi, sm in enumerate(order):
                v = mapping.get(sm)
                axa.add_patch(mpatches.Rectangle(
                    (xi, 0), 1, 1,
                    facecolor=cols.get(str(v), "#c9c9c9"),
                    edgecolor="white", linewidth=0.3))
            axa.set_xlim(0, n); axa.set_ylim(0, 1)
            axa.set_yticks([]); axa.set_xticks([])
            axa.set_ylabel(name, fontsize=_fs(7), rotation=0, ha="right",
                           va="center", labelpad=6)
            axa.spines[:].set_visible(False)
            ann_handles += [mpatches.Patch(facecolor=cols[lv], label=f"{name}: {lv}")
                            for lv in levels]

        ax_h = fig.add_subplot(gs[1 + n_ann, 0])
        im = ax_h.imshow(cons_df.loc[order, order].to_numpy(), cmap="RdYlBu_r",
                         vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax_h.set_xticks(range(n)); ax_h.set_xticklabels(order, rotation=90, fontsize=_fs(6))
        ax_h.set_yticks(range(n)); ax_h.set_yticklabels(order, fontsize=_fs(6))
        for sp in ax_h.spines.values():
            sp.set_visible(False)

        # cluster strip down the right-hand side
        ax_r = fig.add_subplot(gs[1 + n_ann, 1])
        lv_levels = sorted(set(label_map.values()))
        lv_col = {lv: palette(j % 10) for j, lv in enumerate(lv_levels)}
        for yi, sm in enumerate(order):
            ax_r.add_patch(mpatches.Rectangle((0, yi), 1, 1,
                                              facecolor=lv_col[label_map[sm]],
                                              edgecolor="white", linewidth=0.3))
        ax_r.set_xlim(0, 1); ax_r.set_ylim(n, 0)
        ax_r.set_xticks([]); ax_r.set_yticks([])
        ax_r.spines[:].set_visible(False)

        cb = fig.colorbar(im, ax=ax_r, fraction=0.6, pad=0.55)
        cb.set_label("co-clustering frequency", fontsize=_fs(8))
        cb.ax.tick_params(labelsize=_fs(6))

        handles = [mpatches.Patch(facecolor=lv_col[lv], label=lv) for lv in lv_levels]
        ax_h.legend(handles=handles + ann_handles, fontsize=_fs(7), frameon=False,
                    loc="upper left", bbox_to_anchor=(1.12, 1.0))

        mean_off = float(cons_df.to_numpy()[np.triu_indices(n, k=1)].mean())
        fig.suptitle(f"{title}  (n={n}, {X.shape[1]:,} regions, "
                     f"{n_iter} resamples at {subsample_frac:.0%}, k={k})",
                     fontsize=_fs(12), y=0.98)
        ax_h.set_xlabel(f"mean off-diagonal consensus {mean_off:.2f}"
                        f"  ·  1.0 = always together, 0.0 = never",
                        fontsize=_fs(7))
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "methylation_consensus.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        print(summary.round(3).to_string(index=False))
        return {"consensus": cons_df, "labels": label_map, "order": order,
                "summary": summary}

    @staticmethod
    def plot_methylation_heatmap(
        beta: pd.DataFrame,
        top_n_var: int = 2000,
        n_clusters: int = 3,
        annotations: "dict | None" = None,
        cmap: str = "RdYlBu_r",
        title: str = "Chordoma — methylation heatmap",
        save_dir: "str | None" = None,
    ) -> dict:
        """
        Hierarchically clustered beta-value heatmap (regions x samples) with a
        sample dendrogram and optional annotation strips.

        Returns {sample: cluster_label}, same labels as plot_methylation_umap
        when called with the same top_n_var / n_clusters.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if beta is None or beta.empty:
            print("[INFO] No beta matrix — skipping methylation heatmap.")
            return {}

        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list

        X = CohortMethods._top_variable(beta, top_n_var)         # regions x samples
        samples = list(X.columns)
        if len(samples) < 3:
            print(f"[WARN] Only {len(samples)} samples with methylation — "
                  "clustering needs at least 3.")
            return {}

        n_clusters = min(n_clusters, len(samples))
        Zc = linkage(X.T.to_numpy(), method="ward")
        Zr = linkage(X.to_numpy(),   method="ward")
        col_order = leaves_list(Zc)
        row_order = leaves_list(Zr)
        labels = fcluster(Zc, t=n_clusters, criterion="maxclust")
        cluster_map = {s: f"MC{l}" for s, l in zip(samples, labels)}

        M = X.to_numpy()[np.ix_(row_order, col_order)]
        ordered = [samples[i] for i in col_order]

        n_ann = 1 + (len(annotations) if annotations else 0)
        fig = plt.figure(figsize=(max(9, len(samples) * 0.32), 9))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, n_ann * 0.28, 8],
                              width_ratios=[24, 1], hspace=0.02, wspace=0.02)
        ax_dend = fig.add_subplot(gs[0, 0])
        ax_ann  = fig.add_subplot(gs[1, 0])
        ax_map  = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[2, 1])

        dendrogram(Zc, ax=ax_dend, no_labels=True, color_threshold=0,
                   link_color_func=lambda _: "#444444")
        ax_dend.set_xticks([])
        ax_dend.set_yticks([])
        ax_dend.spines[:].set_visible(False)

        palette = plt.get_cmap("tab10")
        tracks = {"Methylation cluster": cluster_map}
        tracks.update(annotations or {})
        ax_ann.set_xlim(0, len(ordered))
        ax_ann.set_ylim(0, len(tracks))
        ax_ann.invert_yaxis()
        legend_handles = []
        for i, (name, mapping) in enumerate(tracks.items()):
            lv = sorted({str(v) for v in mapping.values() if v is not None})
            cols = {v: palette(j % 10) for j, v in enumerate(lv)}
            for c, s in enumerate(ordered):
                v = mapping.get(s)
                ax_ann.add_patch(mpatches.Rectangle(
                    (c, i), 1, 1,
                    facecolor=cols.get(str(v), "#ffffff") if v is not None else "#ffffff",
                    edgecolor="white", linewidth=0.3))
            legend_handles += [mpatches.Patch(facecolor=cols[v], label=f"{name}: {v}")
                               for v in lv]
        ax_ann.set_yticks(np.arange(len(tracks)) + 0.5)
        ax_ann.set_yticklabels(list(tracks.keys()), fontsize=_fs(7))
        ax_ann.set_xticks([])
        ax_ann.tick_params(length=0)
        ax_ann.spines[:].set_visible(False)

        im = ax_map.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                           interpolation="nearest")
        ax_map.set_xticks(range(len(ordered)))
        ax_map.set_xticklabels([SNVmethods._sample_label(s) for s in ordered],
                               rotation=90, fontsize=_fs(7))
        ax_map.set_yticks([])
        ax_map.set_ylabel(f"{M.shape[0]:,} most variable regions")
        fig.colorbar(im, cax=ax_cbar).set_label("beta (fraction methylated)", fontsize=_fs(8))

        if legend_handles:
            ax_map.legend(handles=legend_handles, fontsize=_fs(7), loc="upper left",
                          bbox_to_anchor=(1.06, 1.0), frameon=False)

        fig.suptitle(f"{title}  (n={len(samples)} samples, ward linkage, k={n_clusters})",
                     fontsize=_fs(12), y=0.94)

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, "methylation_heatmap.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        return cluster_map

    # ── co-occurrence ────────────────────────────────────────────────────────

    @staticmethod
    def plot_cohort_co_occurrence(
        alt: pd.DataFrame,
        genes: "list | None" = None,
        min_altered: int = 2,
        title: str = "Gene co-alteration (Fisher exact)",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        Pairwise co-occurrence / mutual exclusivity across the cohort.
        Cells show signed -log10(p) from a Fisher exact test: red = tend to
        co-occur, blue = tend to be mutually exclusive.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        from scipy.stats import fisher_exact

        if alt is None or alt.empty:
            print("[INFO] No alteration matrix — skipping co-occurrence plot.")
            return

        wide = alt.pivot_table(index="sample", columns="gene",
                               values="altered", aggfunc="any").fillna(False)
        genes = [g for g in (genes or list(wide.columns))
                 if g in wide.columns and wide[g].sum() >= min_altered]
        if len(genes) < 2:
            print(f"[INFO] Fewer than two genes altered in ≥{min_altered} samples "
                  "— nothing to correlate.")
            return

        n = len(genes)
        M = np.full((n, n), np.nan)
        for i, gi in enumerate(genes):
            for j, gj in enumerate(genes):
                if i >= j:
                    continue
                a = int(( wide[gi] &  wide[gj]).sum())
                b = int(( wide[gi] & ~wide[gj]).sum())
                c = int((~wide[gi] &  wide[gj]).sum())
                d = int((~wide[gi] & ~wide[gj]).sum())
                odds, p = fisher_exact([[a, b], [c, d]])
                sign = 1.0 if odds > 1 else -1.0
                val = sign * -np.log10(max(p, 1e-12))
                M[i, j] = M[j, i] = val

        fig, ax = plt.subplots(figsize=(1 + n * 0.55, 1 + n * 0.55))
        lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(n)); ax.set_xticklabels(genes, rotation=90, fontsize=_fs(8), style="italic")
        ax.set_yticks(range(n)); ax.set_yticklabels(genes, fontsize=_fs(8), style="italic")
        for i in range(n):
            for j in range(n):
                if i < j and np.isfinite(M[i, j]) and abs(M[i, j]) > -np.log10(0.05):
                    ax.text(j, i, "*", ha="center", va="center", fontsize=_fs(11))
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("signed -log10(p)   red = co-occur, blue = exclusive", fontsize=_fs(7))
        ax.set_title(f"{title}\n(* p < 0.05, n={len(wide)} samples)", fontsize=_fs(10))

        plt.tight_layout()
        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "cohort_co_occurrence.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return pd.DataFrame(M, index=genes, columns=genes)


##########################################################
#### IGV-STYLE READ-LEVEL PANELS                       ####
##########################################################

# IGV's nucleotide colours, used for mismatch ticks and the coverage track
BASE_COLORS: dict = {
    "A": "#00A000",   # green
    "C": "#0000EE",   # blue
    "G": "#C8A000",   # amber
    "T": "#EE0000",   # red
    "N": "#999999",
}

READ_GREY      = "#C8C8C8"   # ordinary read body
READ_GREY_EDGE = "#9A9A9A"
DEL_LINE       = "#333333"   # deletion gap within a read
INS_MARK       = "#8B008B"   # insertion tick


class IGVmethods:
    """
    IGV-style read-level figures drawn straight from a BAM with pysam.

    plot_igv_read_panels is the entry point: one panel per locus, so a
    translocation or fusion can be shown as two panels side by side with the
    split reads that support it coloured consistently in both — the layout of a
    typical IGV fusion screenshot.

    Each panel carries
      · a coverage histogram, coloured at positions where a fraction of reads
        disagree with the reference (IGV's allele-fraction colouring)
      · the reads themselves: split/supplementary reads on top in colour,
        ordinary reads below in grey
      · per-read mismatches as coloured ticks, deletions as gaps, insertions
        as ticks
      · an optional gene model track from a GENCODE GTF

    Mismatches are read from the BAM's MD tag, so no reference FASTA is needed.
    Because ONT reads carry a high raw error rate, only positions where at
    least `min_snv_frac` of the covering reads disagree are drawn — set it to 0
    for the full, noisy IGV view.
    """

    # ── region handling ──────────────────────────────────────────────────────

    @staticmethod
    def parse_region(region, flank_bp: int = 0) -> tuple:
        """
        Accept "chr7:55,019,017-55,211,628", ("7", start, end), a gene name from
        GENE_COORDS / CHORDOMA_GENE_COORDS, or {"chrom":..,"start":..,"end":..}.
        Returns (chrom_without_prefix, start, end, label).

        `flank_bp` widens the window on both sides. It applies to every region
        form, so an explicit span is padded too — pass 0 to get exactly what was
        asked for. For a gene name the label records the flank, because a gene
        name alone no longer describes the window being drawn.
        """
        def _pad(c, s, e, label, gene=None):
            s2 = max(1, int(s) - flank_bp)
            e2 = int(e) + flank_bp
            if gene and flank_bp:
                label = f"{gene}  ±{flank_bp/1e6:g} Mb"
            return c, s2, e2, label

        if isinstance(region, (tuple, list)) and len(region) == 3:
            c, s, e = region
            c = CohortMethods._norm_chrom(c)
            return _pad(c, s, e, f"chr{c}:{int(s):,}-{int(e):,}")

        if isinstance(region, str) and ":" in region:
            chrom, span = region.split(":", 1)
            s, e = span.replace(",", "").split("-")
            c = CohortMethods._norm_chrom(chrom)
            return _pad(c, s, e, f"chr{c}:{int(s):,}-{int(e):,}")

        # gene name or coord dict
        name, coords = CohortMethods.resolve_gene(region)
        c = CohortMethods._norm_chrom(coords["chrom"])
        return _pad(c, coords["start"], coords["end"], name, gene=name)

    @staticmethod
    def gene_span(region):
        """
        (chrom, start, end) of the GENE BODY for a gene-name region, else None.

        parse_region returns the padded window, which loses where the gene
        itself is. With a Mb-scale flank the gene is a fraction of a percent of
        the view, so every locus plot marks its borders — this is what they mark.
        """
        if isinstance(region, (tuple, list)) or (isinstance(region, str)
                                                 and ":" in region):
            return None
        try:
            _, coords = CohortMethods.resolve_gene(region)
        except (KeyError, TypeError):
            return None
        return (CohortMethods._norm_chrom(coords["chrom"]),
                int(coords["start"]), int(coords["end"]))

    @staticmethod
    def _mark_gene(ax, span, chrom, color="#b8860b", lw=1.0):
        """Dashed verticals at the gene's own borders, on any locus axis."""
        if not span or span[0] != chrom:
            return
        for x in (span[1], span[2]):
            ax.axvline(x, color=color, lw=lw, ls="--", alpha=0.85, zorder=4)

    @staticmethod
    def _bam_chrom(bam, chrom: str) -> str:
        """Match the BAM's naming convention (chr7 vs 7)."""
        refs = set(bam.references)
        for cand in (f"chr{chrom}", chrom):
            if cand in refs:
                return cand
        raise KeyError(f"Chromosome {chrom!r} not in BAM "
                       f"(first refs: {list(bam.references)[:3]})")

    # ── SA tag ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sa(read) -> list:
        """
        Supplementary alignments from the SA tag as [(chrom, pos), ...].
        SA format: rname,pos,strand,CIGAR,mapQ,NM;
        """
        if not read.has_tag("SA"):
            return []
        out = []
        for entry in str(read.get_tag("SA")).rstrip(";").split(";"):
            if not entry:
                continue
            f = entry.split(",")
            if len(f) < 2:
                continue
            try:
                out.append((CohortMethods._norm_chrom(f[0]), int(f[1])))
            except ValueError:
                continue
        return out

    # ── read geometry ────────────────────────────────────────────────────────

    @staticmethod
    def _read_blocks(read, min_indel_bp: int = 20) -> tuple:
        """
        Walk the CIGAR and return (blocks, deletions, insertions):
          blocks     [(start, end)]  aligned stretches, reference coordinates
          deletions  [(start, end)]  D/N gaps at least min_indel_bp long
          insertions [(pos, length)] I events at least min_indel_bp long
        Soft/hard clips consume no reference and are skipped.

        Indels shorter than `min_indel_bp` are absorbed into the read body
        rather than drawn. An ONT read carries thousands of 1–2 bp error indels;
        splitting the body at each one turns every read into a stripe pattern
        and buries the structural events worth looking at.
        """
        blocks, dels, ins = [], [], []
        ref = read.reference_start
        cur_start = ref
        cur_len = 0
        for op, ln in (read.cigartuples or []):
            if op in (0, 7, 8):          # M / = / X — consume both
                cur_len += ln
                ref += ln
            elif op in (2, 3):           # D / N — consume reference
                if ln >= min_indel_bp:
                    if cur_len:
                        blocks.append((cur_start, cur_start + cur_len))
                    dels.append((ref, ref + ln))
                    ref += ln
                    cur_start, cur_len = ref, 0
                else:                     # small gap — keep the body continuous
                    cur_len += ln
                    ref += ln
            elif op == 1:                # I — consumes query only
                if ln >= min_indel_bp:
                    ins.append((ref, ln))
            # 4 (S) and 5 (H) consume neither reference nor block
        if cur_len:
            blocks.append((cur_start, cur_start + cur_len))
        return blocks, dels, ins

    @staticmethod
    def _pack_rows(items: list, gap: int) -> list:
        """
        Greedy IGV-style row packing: each read goes in the first row whose last
        read ended more than `gap` bases before it starts. Returns a row index
        per item; `items` must be (start, end, payload) sorted by start.
        """
        row_end: list = []
        rows = []
        for start, end, _ in items:
            placed = False
            for i, e in enumerate(row_end):
                if start > e + gap:
                    row_end[i] = end
                    rows.append(i)
                    placed = True
                    break
            if not placed:
                row_end.append(end)
                rows.append(len(row_end) - 1)
        return rows

    # ── gene models ──────────────────────────────────────────────────────────

    _gtf_cache: dict = {}

    @staticmethod
    def load_gene_models(gtf_path: str, chrom: str) -> pd.DataFrame:
        """
        Exons and gene spans for one chromosome from a GENCODE GTF, cached per
        session (the first call on a chromosome streams the whole file).
        Returns: gene, feature ("gene"|"exon"), start, end, strand
        """
        key = (str(gtf_path), str(chrom))
        if key in IGVmethods._gtf_cache:
            return IGVmethods._gtf_cache[key]

        want = {f"chr{chrom}", str(chrom)}
        name_re = re.compile(r'gene_name "([^"]+)"')
        opener = gzip.open if str(gtf_path).endswith(".gz") else open
        rows = []
        try:
            with opener(gtf_path, "rt") as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.split("\t", 9)
                    if len(f) < 9 or f[0] not in want:
                        continue
                    if f[2] not in ("gene", "exon"):
                        continue
                    m = name_re.search(f[8])
                    if not m:
                        continue
                    rows.append((m.group(1), f[2], int(f[3]), int(f[4]), f[6]))
        except OSError as e:
            print(f"[WARN] Could not read GTF {gtf_path}: {e}")
            return pd.DataFrame(columns=["gene", "feature", "start", "end", "strand"])

        df = pd.DataFrame(rows, columns=["gene", "feature", "start", "end", "strand"])
        IGVmethods._gtf_cache[key] = df
        return df

    @staticmethod
    def _draw_gene_track(ax, gtf_path, chrom, start, end, max_genes: int = 12):
        """Exons as boxes on an intron line, one row per gene."""
        ax.set_xlim(start, end)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=_fs(7))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1e6:,.3f}"))
        ax.set_xlabel(f"chr{chrom} position (Mb)", fontsize=_fs(8))
        if not gtf_path:
            ax.set_ylim(0, 1)
            return

        gm = IGVmethods.load_gene_models(gtf_path, chrom)
        if gm.empty:
            ax.set_ylim(0, 1)
            return

        genes = gm.loc[(gm["feature"] == "gene")
                       & (gm["end"] >= start) & (gm["start"] <= end)]
        # widest genes first — the ones a locus view is usually about
        genes = genes.assign(_w=genes["end"] - genes["start"]) \
                     .sort_values("_w", ascending=False).head(max_genes)
        if genes.empty:
            ax.set_ylim(0, 1)
            return

        rows = IGVmethods._pack_rows(
            [(r.start, r.end, r.gene) for r in
             genes.sort_values("start").itertuples()],
            gap=int((end - start) * 0.12),
        )
        ordered = list(genes.sort_values("start").itertuples())
        n_rows = max(rows) + 1
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()

        for row, g in zip(rows, ordered):
            ax.hlines(row, max(g.start, start), min(g.end, end),
                      color="#2c3e8f", linewidth=1.0, zorder=1)
            ex = gm.loc[(gm["feature"] == "exon") & (gm["gene"] == g.gene)
                        & (gm["end"] >= start) & (gm["start"] <= end)]
            for e in ex.itertuples():
                ax.add_patch(mpatches.Rectangle(
                    (e.start, row - 0.28), max(e.end - e.start, 1), 0.56,
                    facecolor="#2c3e8f", edgecolor="none", zorder=2))
            vis = min(g.end, end) - max(g.start, start)
            if vis < (end - start) * 0.04:      # too narrow to label legibly
                continue
            xm = (max(g.start, start) + min(g.end, end)) / 2
            ax.text(xm, row + 0.44, f"{g.gene} {'▸' if g.strand == '+' else '◂'}",
                    ha="center", va="top", fontsize=_fs(7), color="#2c3e8f")

    # ── the main figure ──────────────────────────────────────────────────────

    @staticmethod
    def plot_igv_read_panels(
        bam_path: str,
        regions: list,
        sample: "str | None" = None,
        flank_bp: int = 0,
        snv_df: "pd.DataFrame | None" = None,
        gtf_path: "str | None" = None,
        max_reads: int = 60,
        min_mapq: int = 1,
        min_snv_frac: float = 0.2,
        min_depth: int = 3,
        min_indel_bp: int = 20,
        show_coverage: bool = True,
        link_split_reads: bool = True,
        link_slop: int = 5_000,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
        figsize: "tuple | None" = None,
    ):
        """
        IGV-style read panels, one column per region.

        Parameters
        ----------
        bam_path  : indexed BAM (a region slice is fine — see the note below)
        regions   : list of loci. Each may be "chr7:55,019,017-55,211,628",
                    ("7", start, end), a gene name known to GENE_COORDS /
                    CHORDOMA_GENE_COORDS, or {"chrom":..,"start":..,"end":..}.
                    Two regions side by side is the fusion / translocation view.
        flank_bp  : widen every window by this much on each side (default 0 =
                    gene body only). Use it to put a locus in context — 2 Mb
                    shows nearby breakpoints and the CNV neighbourhood. Two
                    caveats: (1) a mini-BAM only holds the interval it was
                    sliced with, so asking for more than that flank just draws
                    empty margins — a warning is printed when the reads cover
                    much less of the window than requested; (2) at multi-Mb
                    widths an ONT read is a hairline and per-base mismatches
                    are sub-pixel, so this is an SV/structure view, not an
                    SNV view.
        snv_df    : optional called variants (columns sample/chrom/pos) drawn as
                    dashed vertical guides, so caller output can be compared
                    against what the reads actually show.
        gtf_path  : GENCODE GTF for the gene model track (optional)
        min_snv_frac : only draw a mismatch where at least this fraction of
                    covering reads disagree with the reference. ONT reads carry
                    a high raw error rate, so 0 gives the true IGV view but is
                    very noisy; 0.2 keeps candidate variants.
        min_indel_bp : indels below this size are absorbed into the read body
                    instead of being drawn (default 20 bp). ONT reads carry
                    thousands of 1–2 bp error indels; drawing them all hides the
                    read structure completely.
        link_split_reads : colour reads whose supplementary alignment (SA tag)
                    lands in another panel, consistently across panels — these
                    are the reads supporting the junction.
        link_slop : how far outside a region an SA alignment may fall and still
                    count as linking to it.

        Reads are fetched with supplementary alignments kept (that is how one
        read appears in both panels) and secondary alignments dropped.

        Returns a DataFrame of the linking reads: read_name, panels, chrom, pos.
        """
        try:
            import pysam
        except ImportError:
            print("[WARN] pysam is required for IGV-style read plots.")
            return pd.DataFrame()

        if not os.path.exists(bam_path):
            print(f"[WARN] BAM not found: {bam_path}")
            return pd.DataFrame()
        if not (os.path.exists(bam_path + ".bai")
                or os.path.exists(os.path.splitext(bam_path)[0] + ".bai")):
            print(f"[WARN] No .bai index next to {bam_path} — "
                  f"run `samtools index` first.")
            return pd.DataFrame()

        actual_dir = SNVmethods._gene_save_dir(save_dir)
        parsed = [IGVmethods.parse_region(r, flank_bp=flank_bp) for r in regions]
        spans  = [IGVmethods.gene_span(r) for r in regions]
        n_panels = len(parsed)

        bam = pysam.AlignmentFile(bam_path, "rb")

        # ── pass 1: fetch reads, note which ones link panels ────────────────
        panel_reads = []
        for chrom, start, end, _ in parsed:
            try:
                bchrom = IGVmethods._bam_chrom(bam, chrom)
            except KeyError as e:
                print(f"[WARN] {e}")
                panel_reads.append([])
                continue
            reads = []
            for r in bam.fetch(bchrom, max(start, 0), end):
                if r.is_unmapped or r.is_secondary:
                    continue
                if r.mapping_quality < min_mapq:
                    continue
                reads.append(r)
            panel_reads.append(reads)

            # A mini-BAM only contains the interval it was sliced with. If the
            # requested window is much wider, the extra margin is empty on both
            # sides and looks like real loss of coverage — say so instead.
            if reads and (end - start) > 0:
                lo = min(r.reference_start for r in reads)
                hi = max(r.reference_end or r.reference_start for r in reads)
                covered = (min(hi, end) - max(lo, start)) / (end - start)
                if covered < 0.9:
                    print(f"[WARN] chr{chrom}:{start:,}-{end:,}: reads only span "
                          f"{lo:,}-{hi:,} ({covered:.0%} of the window). If this "
                          f"is a mini-BAM, re-slice it with a flank at least as "
                          f"wide as flank_bp.")

        # which read names bridge two panels?
        link_panels: dict = defaultdict(set)
        if link_split_reads:
            for pi, reads in enumerate(panel_reads):
                for r in reads:
                    link_panels[r.query_name].add(pi)
                    for sc, sp in IGVmethods._parse_sa(r):
                        for pj, (c2, s2, e2, _) in enumerate(parsed):
                            if sc == c2 and (s2 - link_slop) <= sp <= (e2 + link_slop):
                                link_panels[r.query_name].add(pj)
        linked_names = {n for n, ps in link_panels.items() if len(ps) > 1}

        palette = plt.get_cmap("tab20")
        link_color = {n: palette(i % 20) for i, n in enumerate(sorted(linked_names))}

        # ── figure scaffold ─────────────────────────────────────────────────
        height_ratios = ([1.1] if show_coverage else []) + [6.0, 0.9]
        fig_w = figsize[0] if figsize else max(7.0, 6.2 * n_panels)
        fig_h = figsize[1] if figsize else 8.0
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(len(height_ratios), n_panels,
                              height_ratios=height_ratios,
                              hspace=0.06, wspace=0.06)

        link_rows = []
        for pi, (chrom, start, end, label) in enumerate(parsed):
            reads = panel_reads[pi]
            row = 0
            ax_cov = fig.add_subplot(gs[row, pi]) if show_coverage else None
            if show_coverage:
                row += 1
            ax = fig.add_subplot(gs[row, pi])
            ax_gene = fig.add_subplot(gs[row + 1, pi])
            for _a in ([ax_cov] if ax_cov is not None else []) + [ax]:
                IGVmethods._mark_gene(_a, spans[pi], chrom)

            if not reads:
                ax.text(0.5, 0.5, "no reads in region", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_fs(9), color="#888888")
                ax.set_xlim(start, end)
                ax.set_yticks([])
                IGVmethods._draw_gene_track(ax_gene, gtf_path, chrom, start, end)
                ax.set_title(label, fontsize=_fs(10))
                continue

            # ── pass 2: reference bases, depth, per-read mismatches ─────────
            width = end - start
            depth = np.zeros(width + 1, dtype=np.int32)
            nonref = np.zeros(width + 1, dtype=np.int32)
            alt_counts = defaultdict(lambda: defaultdict(int))
            read_mm = {}

            for r in reads:
                mm = []
                try:
                    pairs = r.get_aligned_pairs(matches_only=True, with_seq=True)
                except ValueError:
                    pairs = []          # no MD tag on this read
                seq = r.query_sequence or ""
                for qpos, rpos, refbase in pairs:
                    if rpos is None or not (start <= rpos <= end):
                        continue
                    idx = rpos - start
                    depth[idx] += 1
                    if refbase is None:
                        continue
                    # pysam lowercases the reference base at mismatches
                    if refbase.islower():
                        obs = seq[qpos].upper() if qpos is not None and qpos < len(seq) else "N"
                        nonref[idx] += 1
                        alt_counts[rpos][obs] += 1
                        mm.append((rpos, obs))
                read_mm[id(r)] = mm

            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(depth > 0, nonref / np.maximum(depth, 1), 0.0)
            variant_idx = np.where((frac >= min_snv_frac) & (depth >= min_depth))[0]
            variant_pos = set((variant_idx + start).tolist())

            # ── coverage track ─────────────────────────────────────────────
            if show_coverage:
                xs = np.arange(start, start + width + 1)
                ax_cov.fill_between(xs, depth, step="mid", color="#B0B0B0", linewidth=0)
                for idx in variant_idx:
                    pos = int(idx + start)
                    counts = alt_counts.get(pos, {})
                    if not counts:
                        continue
                    base = max(counts, key=counts.get)
                    ax_cov.vlines(pos, 0, depth[idx],
                                  color=BASE_COLORS.get(base, "#999999"), linewidth=0.9)
                ax_cov.set_xlim(start, end)
                ax_cov.set_ylim(0, max(depth.max() * 1.15, 1))
                ax_cov.set_xticks([])
                ax_cov.tick_params(axis="y", labelsize=_fs(6))
                ax_cov.spines[["top", "right", "bottom"]].set_visible(False)
                if pi == 0:
                    ax_cov.set_ylabel("depth", fontsize=_fs(7))
                ax_cov.set_title(label, fontsize=_fs(10))
            else:
                ax.set_title(label, fontsize=_fs(10))

            # ── order reads: linking reads on top, then the rest ────────────
            linked = [r for r in reads if r.query_name in linked_names]
            others = [r for r in reads if r.query_name not in linked_names]
            if len(others) > max_reads:
                step = max(1, len(others) // max_reads)
                others = others[::step][:max_reads]

            gap = max(int(width * 0.005), 1)
            packed = []
            for group in (linked, others):
                group_sorted = sorted(group, key=lambda r: r.reference_start)
                items = [(r.reference_start, r.reference_end or r.reference_start, r)
                         for r in group_sorted]
                packed.append((items, IGVmethods._pack_rows(items, gap)))

            n_link_rows = (max(packed[0][1]) + 1) if packed[0][1] else 0
            sep = 0.6 if n_link_rows else 0.0
            n_other_rows = (max(packed[1][1]) + 1) if packed[1][1] else 0
            total_rows = n_link_rows + sep + n_other_rows

            # ── draw reads ─────────────────────────────────────────────────
            for gi, (items, rows) in enumerate(packed):
                offset = 0 if gi == 0 else n_link_rows + sep
                for (rstart, rend, r), row_i in zip(items, rows):
                    y = offset + row_i
                    is_linked = r.query_name in linked_names
                    color = link_color[r.query_name] if is_linked else READ_GREY
                    edge = "none" if is_linked else READ_GREY_EDGE

                    blocks, dels, ins = IGVmethods._read_blocks(r, min_indel_bp)
                    for b0, b1 in blocks:
                        if b1 < start or b0 > end:
                            continue
                        ax.add_patch(mpatches.Rectangle(
                            (max(b0, start), y - 0.34),
                            max(min(b1, end) - max(b0, start), 1), 0.68,
                            facecolor=color, edgecolor=edge, linewidth=0.5, zorder=2))
                    for d0, d1 in dels:
                        if d1 < start or d0 > end:
                            continue
                        ax.hlines(y, max(d0, start), min(d1, end),
                                  color=DEL_LINE, linewidth=1.0, zorder=3)
                    for ipos, ilen in ins:
                        if start <= ipos <= end:
                            ax.vlines(ipos, y - 0.40, y + 0.40, color=INS_MARK,
                                      linewidth=min(0.8 + ilen / 500.0, 2.4), zorder=4)
                    for rpos, obs in read_mm.get(id(r), []):
                        if rpos in variant_pos:
                            ax.vlines(rpos, y - 0.34, y + 0.34,
                                      color=BASE_COLORS.get(obs, "#999999"),
                                      linewidth=0.9, zorder=5)

            if n_link_rows:
                ax.axhspan(-0.6, n_link_rows - 0.4, facecolor="#fbfbfb",
                           edgecolor="none", zorder=0)
                ax.axhline(n_link_rows + sep / 2 - 0.5, color="#cccccc",
                           linewidth=0.8, linestyle="-")

            # called variants as dashed guides
            if snv_df is not None and not snv_df.empty:
                s = snv_df
                if sample is not None and "sample" in s.columns:
                    s = s.loc[s["sample"] == sample]
                s = s.loc[(s["chrom"].map(CohortMethods._norm_chrom) == chrom)
                          & (s["pos"] >= start) & (s["pos"] <= end)]
                for p in s["pos"].unique():
                    ax.axvline(p, color="#444444", linestyle=":", linewidth=0.8,
                               alpha=0.8, zorder=1)

            ax.set_xlim(start, end)
            ax.set_ylim(-1, max(total_rows, 1))
            ax.invert_yaxis()
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            if pi == 0:
                ax.set_ylabel(f"reads (≤{max_reads} shown)", fontsize=_fs(8))

            IGVmethods._draw_gene_track(ax_gene, gtf_path, chrom, start, end)

            for name in sorted(linked_names):
                if pi in link_panels[name]:
                    link_rows.append({"read_name": name,
                                      "panel": pi, "region": label})

        # ── legend and title ───────────────────────────────────────────────
        handles = [mpatches.Patch(facecolor=READ_GREY, edgecolor=READ_GREY_EDGE,
                                  label="read")]
        if linked_names:
            handles.append(mpatches.Patch(
                facecolor=palette(0),
                label=f"split read linking panels (n={len(linked_names)})"))
        handles += [mlines.Line2D([], [], color=DEL_LINE,
                                  label=f"deletion ≥{min_indel_bp} bp"),
                    mlines.Line2D([], [], color=INS_MARK,
                                  label=f"insertion ≥{min_indel_bp} bp")]
        handles += [mpatches.Patch(facecolor=BASE_COLORS[b], label=f"mismatch {b}")
                    for b in "ACGT"]
        if snv_df is not None and not snv_df.empty:
            handles.append(mlines.Line2D([], [], color="#444444", linestyle=":",
                                         label="called variant"))
        fig.legend(handles=handles, fontsize=_fs(7), frameon=False,
                   loc="lower center", ncol=min(len(handles), 5),
                   bbox_to_anchor=(0.5, -0.06))

        head = title or (f"{sample} — " if sample else "") + "read-level view"
        fig.suptitle(f"{head}   (mismatches shown at ≥{min_snv_frac:.0%} of reads, "
                     f"depth ≥{min_depth})", fontsize=_fs(11), y=0.98)

        bam.close()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "igv_read_panels.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        out = pd.DataFrame(link_rows)
        if not out.empty:
            n = out["read_name"].nunique()
            print(f"{n} read(s) span more than one panel "
                  f"({len(parsed)} panels, SA slop {link_slop:,} bp)")
        elif link_split_reads and len(parsed) > 1:
            print("No split reads link these panels — if a junction is expected, "
                  "widen the regions or lower min_mapq.")
        return out

    @staticmethod
    def plot_sv_breakpoint_reads(
        bam_path: str,
        sv_row: "pd.Series | dict",
        flank_bp: int = 5_000,
        **kwargs,
    ):
        """
        Convenience wrapper: given one row of an SV table (chrom/start/chrom2/end
        /svtype), draw the two breakpoint neighbourhoods side by side.

        For a BND the two panels are (chrom, start) and (chrom2, end); for a
        DEL/DUP/INV they are the two ends of the same event.
        """
        r = sv_row if isinstance(sv_row, dict) else sv_row.to_dict()
        c1 = CohortMethods._norm_chrom(r["chrom"])
        p1 = int(r["start"])
        c2 = CohortMethods._norm_chrom(r.get("chrom2", r["chrom"]))
        p2 = int(r["end"])
        svtype = r.get("svtype", "SV")

        regions = [(c1, p1 - flank_bp, p1 + flank_bp),
                   (c2, p2 - flank_bp, p2 + flank_bp)]
        kwargs.setdefault(
            "title",
            f"{svtype}  chr{c1}:{p1:,} ↔ chr{c2}:{p2:,}"
            + (f"  (AF {r['af']:.2f})" if r.get("af") == r.get("af") and "af" in r else "")
        )
        return IGVmethods.plot_igv_read_panels(bam_path, regions, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# ReportMethods — gather figures scattered across output folders into one page

    # ── locus depth ──────────────────────────────────────────────────────────

    @staticmethod
    def _locus_depth(bam_path, chrom, start, end, bin_bp, min_mapq=1):
        """
        Binned depth over a window from one BAM: (bin_starts, depth, n_reads).

        Depth is accumulated from alignment spans rather than a per-base pileup —
        two orders of magnitude faster over a 4 Mb window and indistinguishable
        once binned at kb scale.
        """
        import pysam
        n_bins = max(1, int(np.ceil((end - start) / bin_bp)))
        cov = np.zeros(n_bins, dtype=float)
        n_reads = 0
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            ref = IGVmethods._bam_chrom(bam, chrom)
            for r in bam.fetch(ref, max(start, 0), end):
                if r.is_unmapped or r.is_secondary or r.is_supplementary:
                    continue
                if r.mapping_quality < min_mapq:
                    continue
                n_reads += 1
                a = max(r.reference_start, start)
                b = min(r.reference_end or r.reference_start, end)
                if b <= a:
                    continue
                b0, b1 = int((a - start) // bin_bp), int((b - start - 1) // bin_bp)
                for i in range(b0, min(b1, n_bins - 1) + 1):
                    lo = start + i * bin_bp
                    hi = lo + bin_bp
                    cov[i] += (min(b, hi) - max(a, lo)) / bin_bp
        return start + np.arange(n_bins) * bin_bp, cov, n_reads

    @staticmethod
    def plot_locus_depth_stack(
        bam_paths: dict,
        region="TBXT",
        flank_bp: int = 2_000_000,
        bin_bp: int = 5_000,
        meta: "pd.DataFrame | None" = None,
        normalise: bool = True,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        One depth row per sample across a locus — the cohort view of coverage.

        bam_paths : {label: path to an indexed BAM}. Mini-BAM slices are fine as
                    long as they were cut with at least `flank_bp`.
        normalise : divide each row by its own median depth in the window, so
                    rows compare as relative copy state instead of as yield. Turn
                    it off to see raw depth (and therefore sequencing depth
                    differences) instead.

        Returns a per-sample summary (n_reads, mean and median depth).
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        try:
            import pysam  # noqa: F401
        except ImportError:
            print("[WARN] pysam is required for depth plots.")
            return pd.DataFrame()

        chrom, start, end, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
        rows, profiles = [], {}
        for lab, path in bam_paths.items():
            if not path or not os.path.exists(path):
                continue
            try:
                xs, cov, n_reads = IGVmethods._locus_depth(path, chrom, start,
                                                           end, bin_bp)
            except (KeyError, ValueError) as e:
                print(f"[WARN] {lab}: {e}")
                continue
            profiles[lab] = (xs, cov)
            rows.append((lab, n_reads, float(cov.mean()), float(np.median(cov))))

        if not profiles:
            print("[WARN] No readable BAMs — nothing to plot.")
            return pd.DataFrame()
        summary = pd.DataFrame(rows, columns=["sample", "n_reads",
                                              "mean_depth", "median_depth"])

        labels = list(profiles)
        n = len(labels)
        fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                                 figsize=(13, max(2.5, 0.45 * n + 1.5)),
                                 squeeze=False)
        axes = axes[:, 0]
        gene_start = gene_end = None
        try:
            _, gc = CohortMethods.resolve_gene(region)
            gene_start, gene_end = int(gc["start"]), int(gc["end"])
        except (KeyError, TypeError):
            pass

        for ax, lab in zip(axes, labels):
            xs, cov = profiles[lab]
            y = cov / max(np.median(cov), 1e-9) if normalise else cov
            ax.fill_between(xs, y, step="mid", color="#2c7fb8", alpha=0.55,
                            linewidth=0)
            if gene_start is not None:
                ax.axvspan(gene_start, gene_end, color="#f2c14e", alpha=0.35,
                           zorder=0, linewidth=0)
                for xb in (gene_start, gene_end):
                    ax.axvline(xb, color="#b8860b", lw=0.9, ls="--", alpha=0.85)
            if normalise:
                ax.axhline(1.0, color="#888888", lw=0.5, ls=":")
            ax.set_ylabel(lab, fontsize=_fs(7), rotation=0, ha="right", va="center",
                          labelpad=5)
            ax.tick_params(axis="y", labelsize=_fs(6))
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylim(0, 3.0 if normalise else None)
        axes[-1].set_xlim(start, end)
        axes[-1].set_xlabel(f"chr{chrom} position (bp)", fontsize=_fs(9))
        fig.suptitle(title or f"Depth across {rlabel}"
                              + ("  (each row / its own median)" if normalise else ""),
                     fontsize=_fs(12), y=0.997)
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "locus_depth_stack.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return summary

    @staticmethod
    def plot_depth_vs_reads(
        summary: pd.DataFrame,
        meta: "pd.DataFrame | None" = None,
        region_label: str = "TBXT ±2 Mb",
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        Mean depth in the window against the number of reads in it, one point
        per sample — the QC counterpart to the stacked profile.

        Points should sit on a straight line through the origin whose slope is
        the mean read length. A sample **above** the line has more depth than
        its read count explains, i.e. shorter reads or a genuine copy gain at
        the locus; **below** means longer reads or a loss. Colour is the
        aggressive/indolent split when `meta` is supplied.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if summary is None or summary.empty:
            print("[INFO] Nothing to plot.")
            return

        grp = (meta["aggressive"].reindex(summary["sample"]).to_numpy()
               if meta is not None and "aggressive" in meta.columns else None)
        colors = ["#c1272d" if g is True else "#2c7fb8" if g is False else "#c9c9c9"
                  for g in (grp if grp is not None else [None] * len(summary))]

        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        ax.scatter(summary["n_reads"], summary["mean_depth"], s=42, c=colors,
                   alpha=0.85, linewidths=0)
        for _, r in summary.iterrows():
            ax.annotate(r["sample"], (r["n_reads"], r["mean_depth"]),
                        fontsize=_fs(6), xytext=(3, 3), textcoords="offset points",
                        color="#555555")
        if len(summary) > 2:
            x = summary["n_reads"].to_numpy(float)
            y = summary["mean_depth"].to_numpy(float)
            slope = float(np.sum(x * y) / max(np.sum(x * x), 1e-9))
            xs = np.linspace(0, x.max() * 1.05, 20)
            ax.plot(xs, slope * xs, color="#888888", lw=0.9, ls="--",
                    label=f"depth = {slope:.2g} x reads")
            ax.legend(fontsize=_fs(7), frameon=False)

        ax.set_xlabel(f"reads in {region_label}", fontsize=_fs(9))
        ax.set_ylabel(f"mean depth in {region_label}", fontsize=_fs(9))
        ax.spines[["top", "right"]].set_visible(False)
        if grp is not None:
            ax.add_artist(ax.legend(handles=[
                mpatches.Patch(color="#c1272d", label="aggressive"),
                mpatches.Patch(color="#2c7fb8", label="indolent"),
                mpatches.Patch(color="#c9c9c9", label="no metadata")],
                fontsize=_fs(7), frameon=False, loc="lower right"))
        fig.suptitle(title or f"Coverage vs reads — {region_label}", fontsize=_fs(12))
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "depth_vs_reads.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

# ═════════════════════════════════════════════════════════════════════════════

class ReportMethods:
    """
    Collect the figures written by the cells above (plus anything produced
    outside the notebook, e.g. the modkit plots folder) into a single browsable
    HTML report.
    """

    FIGURE_EXT = (".pdf", ".png", ".svg", ".jpg", ".jpeg")
    TABLE_EXT  = (".tsv", ".csv", ".txt")
    PAGE_EXT   = (".html", ".htm")

    @staticmethod
    def _slug(text: str) -> str:
        keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
        return "".join(keep).strip("_") or "section"

    @staticmethod
    def _natural_key(path: str):
        """Sort c1-p2 before c1-p10, and TBXT_2000kb after TBXT_50kb sensibly."""
        parts = re.split(r"(\d+)", os.path.basename(path).lower())
        return [int(p) if p.isdigit() else p for p in parts]

    @staticmethod
    def _collect(source: str, include_tables: bool, include_pages: bool) -> list:
        """Every figure/table under `source`, as (relative_path, abs_path)."""
        exts = ReportMethods.FIGURE_EXT
        if include_tables:
            exts = exts + ReportMethods.TABLE_EXT
        if include_pages:
            exts = exts + ReportMethods.PAGE_EXT

        if os.path.isfile(source):
            return [(os.path.basename(source), source)]

        out = []
        for dirpath, dirnames, filenames in os.walk(source):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if not fn.lower().endswith(exts):
                    continue
                ap = os.path.join(dirpath, fn)
                out.append((os.path.relpath(ap, source), ap))
        return sorted(out, key=lambda t: (os.path.dirname(t[0]),
                                          ReportMethods._natural_key(t[0])))

    @staticmethod
    def build_plot_report(
        sources: dict,
        out_dir: str,
        title: str = "Chordoma cohort report",
        subtitle: "str | None" = None,
        copy_files: bool = True,
        include_tables: bool = True,
        include_pages: bool = True,
        embed_height: int = 620,
        max_embeds_open: int = 12,
    ) -> "str | None":
        """
        Write `{out_dir}/index.html` linking every figure found under `sources`.

            ReportMethods.build_plot_report(
                sources = {
                    "Cohort overview": PLOT_BASE,
                    "Methylation":     f"{MDA}/modkit/ONTWGS10/plots",
                },
                out_dir = f"{MDA}/cordoma_plots_20260902_1030",
            )

        `sources` maps a section heading to a directory (walked recursively) or
        a single file. Subdirectories become collapsible subsections, so the
        modkit tree (browser/, dmr_heatmap/, locus_volcano/, ...) keeps its
        shape instead of collapsing into one long list.

        copy_files=True (default) copies everything into `out_dir`, giving a
        self-contained folder you can move or zip. False leaves files where they
        are and links relatively — smaller, but the report breaks if either side
        moves.

        PDFs are shown in lazy <iframe>s: only the first `max_embeds_open`
        subsections start expanded, because a few hundred embedded PDFs will
        stall a browser. Everything else is one click away.

        Returns the path to index.html, or None if nothing was found.
        """
        os.makedirs(out_dir, exist_ok=True)
        index = os.path.join(out_dir, "index.html")
        # writing the report into a folder that is also a source is the normal
        # case (copy_files=False), so never let a previous run's index.html be
        # collected and embedded inside the new one
        skip_abs = os.path.abspath(index)

        sections = []
        n_files = 0
        for heading, source in sources.items():
            if not source or not os.path.exists(source):
                print(f"[INFO] {heading}: not found ({source}) — skipped.")
                continue
            found = ReportMethods._collect(source, include_tables, include_pages)
            found = [(rel, ap) for rel, ap in found
                     if os.path.abspath(ap) != skip_abs]
            if not found:
                print(f"[INFO] {heading}: no figures under {source} — skipped.")
                continue

            sec_slug = ReportMethods._slug(heading)
            groups: dict = {}
            for rel, ap in found:
                sub = os.path.dirname(rel) or "."
                if copy_files:
                    dest = os.path.join(out_dir, sec_slug, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(ap, dest)
                    href = os.path.relpath(dest, out_dir)
                else:
                    href = os.path.relpath(ap, out_dir)
                groups.setdefault(sub, []).append((os.path.basename(rel), href))
                n_files += 1

            sections.append((heading, sec_slug, groups,
                             sum(len(v) for v in groups.values())))
            print(f"  {heading}: {sum(len(v) for v in groups.values())} files "
                  f"in {len(groups)} group(s)")

        if not sections:
            print("[WARN] Nothing to report — no source directory had figures.")
            return None

        html = ReportMethods._render(sections, title, subtitle, n_files,
                                     embed_height, max_embeds_open)
        with open(index, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"\nReport: {index}")
        print(f"  {n_files} files across {len(sections)} section(s)"
              + ("  (copied — folder is self-contained)" if copy_files
                 else "  (linked in place — do not move either folder)"))
        return index

    # ── rendering ────────────────────────────────────────────────────────────

    @staticmethod
    def _render(sections, title, subtitle, n_files, embed_height,
                max_embeds_open) -> str:
        esc = html_lib.escape
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        toc = "\n".join(
            f'<li><a href="#{s}">{esc(h)}</a> <span class="n">{n}</span></li>'
            for h, s, _, n in sections
        )

        body, opened = [], 0
        for heading, sec_slug, groups, n_sec in sections:
            body.append(f'<section id="{sec_slug}"><h2>{esc(heading)}'
                        f' <span class="n">{n_sec}</span></h2>')
            for sub, items in groups.items():
                sub_label = "." if sub == "." else sub
                opened += 1
                is_open = " open" if opened <= max_embeds_open else ""
                body.append(
                    f'<details{is_open}><summary>{esc(sub_label)}'
                    f' <span class="n">{len(items)}</span></summary><div class="grid">'
                )
                for name, href in items:
                    low = name.lower()
                    if low.endswith(ReportMethods.PAGE_EXT):
                        body.append(
                            f'<figure class="wide"><figcaption>{esc(name)}</figcaption>'
                            f'<iframe loading="lazy" src="{esc(href)}" '
                            f'height="{embed_height}"></iframe>'
                            f'<a class="dl" href="{esc(href)}" target="_blank">open</a>'
                            f'</figure>')
                    elif low.endswith((".png", ".jpg", ".jpeg", ".svg")):
                        body.append(
                            f'<figure><figcaption>{esc(name)}</figcaption>'
                            f'<a href="{esc(href)}" target="_blank">'
                            f'<img loading="lazy" src="{esc(href)}"></a></figure>')
                    elif low.endswith(".pdf"):
                        body.append(
                            f'<figure><figcaption>{esc(name)}</figcaption>'
                            f'<iframe loading="lazy" src="{esc(href)}" '
                            f'height="{embed_height}"></iframe>'
                            f'<a class="dl" href="{esc(href)}" target="_blank">open</a>'
                            f'</figure>')
                    else:
                        body.append(
                            f'<div class="tbl"><a href="{esc(href)}" '
                            f'target="_blank">{esc(name)}</a></div>')
                body.append("</div></details>")
            body.append("</section>")

        sub_html = f"<p class='sub'>{esc(subtitle)}</p>" if subtitle else ""
        return f"""<!doctype html>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>
 :root {{ --fg:#1a1a1a; --mut:#6a6a6a; --line:#e2e2e2; --bg:#ffffff; --card:#fafafa; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e8e8e8; --mut:#9a9a9a; --line:#333; --bg:#161616; --card:#1e1e1e; }}
 }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--fg);
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        display:grid; grid-template-columns:250px 1fr; }}
 nav {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto;
       border-right:1px solid var(--line); padding:20px 16px; }}
 nav h1 {{ font-size:15px; margin:0 0 4px; }}
 nav ul {{ list-style:none; margin:12px 0 0; padding:0; }}
 nav li {{ margin:2px 0; }}
 nav a {{ color:var(--fg); text-decoration:none; }}
 nav a:hover {{ text-decoration:underline; }}
 main {{ padding:24px 28px 80px; min-width:0; }}
 .sub, .meta {{ color:var(--mut); font-size:12px; }}
 h2 {{ font-size:17px; margin:28px 0 10px; padding-bottom:6px;
      border-bottom:1px solid var(--line); }}
 .n {{ color:var(--mut); font-weight:400; font-size:11px; }}
 details {{ margin:10px 0; }}
 summary {{ cursor:pointer; padding:6px 0; color:var(--mut);
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
 .grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); }}
 figure {{ margin:0; background:var(--card); border:1px solid var(--line);
          border-radius:6px; padding:10px; min-width:0; }}
 figure.wide {{ grid-column:1/-1; }}
 figcaption {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
              font-size:11px; color:var(--mut); margin-bottom:8px;
              overflow-wrap:anywhere; }}
 iframe {{ width:100%; border:0; background:#fff; border-radius:3px; }}
 img {{ width:100%; height:auto; border-radius:3px; background:#fff; }}
 .dl {{ display:inline-block; margin-top:6px; font-size:11px; color:var(--mut); }}
 .tbl {{ background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding:8px 10px; font-family:ui-monospace,Menlo,monospace; font-size:11px;
        overflow-wrap:anywhere; }}
 .tbl a {{ color:var(--fg); }}
 @media (max-width:900px) {{ body {{ grid-template-columns:1fr; }}
   nav {{ position:static; height:auto; border-right:0;
         border-bottom:1px solid var(--line); }} }}
</style>
<nav>
 <h1>{esc(title)}</h1>
 {sub_html}
 <p class="meta">{stamp}<br>{n_files} files</p>
 <ul>{toc}</ul>
</nav>
<main>
{chr(10).join(body)}
</main>
"""


# ═════════════════════════════════════════════════════════════════════════════
# PatientMethods — sample↔patient metadata, aggressiveness, group comparison
# ═════════════════════════════════════════════════════════════════════════════

# Known errors in the clinical sheets, applied by load_patient_meta and
# load_survival_meta so every figure sees the same corrected value. This is a
# stopgap for the source CSVs, not a place to encode judgement calls: each
# entry must be a confirmed data-entry error, and should be deleted once the
# CSV itself is fixed. Every applied fix is printed, so it can never go silent.
SHEET_FIXES: dict[str, dict] = {
    # ONTWGS12-54 (P18) records Recurrence=N with a count of 0, yet carries a
    # dated recurrence (2022-12-29). Confirmed a workbook entry error: the
    # patient had one recurrence.
    "ONTWGS12-54-240567-NP01": {"n_recurrence": 1},
}


# Patients whose RECURRENCE TIMING cannot be trusted, keyed by patient id with
# the reason. Consumed by PatientMethods.recurrence_timing_class, which marks
# them "unknown" so they fall out of the aggressive/indolent split.
#
# SCOPED TO RECURRENCE TIMING ONLY, deliberately. Death and metastasis are
# recorded for every one of these patients, so they still count in full under
# the outcome-weighted schemes; it is only "how fast did it come back" that is
# unanswerable for them. Dropping them from the cohort outright would discard
# sound data.
#
# All seven recur but carry no recurrence date. PFS_months does hold the
# interval for them (it matches the date-derived value exactly, r = 1.0000,
# wherever both exist), so recurrence_timing_class can recover the timing with
# use_pfs_fallback=True — but that path is a second-hand source, and listing a
# patient here overrides it. Remove an entry once the date reaches the sheet.
# Corrections applied to the SURVIVAL SHEET
# (Chordoma_molecular_survival_extended+ONTWGS10_blinded.csv), keyed by its
# `sample` value. Same contract as SHEET_FIXES: each entry is a confirmed
# data-entry error, every applied fix is printed, and the entry should be
# deleted once the CSV itself is corrected.
#
# The sheet is authoritative for recurrence — chordoma_sample_qc_full.csv's
# n_recurrences column is not — so a typo here has to be fixed here rather
# than worked around downstream.
# Empty on purpose. ONTWGS12-54 (P18) used to read Recurrence="N" with a count
# of 0 next to a dated recurrence (12/29/22 in three columns); that was fixed
# in the CSV itself on 2026-09-13, so no override is needed. The mechanism is
# kept for the next one.
SURVIVAL_SHEET_FIXES: dict[str, dict] = {}


IGNORED_PATIENTS: dict[str, str] = {
    "P4":  "recurrence recorded, no date (PFS 47.0 mo)",
    "P5":  "recurrence recorded, no date (PFS 37.0 mo)",
    "P10": "recurrence recorded, no date (PFS 21.0 mo)",
    "P17": "recurrence recorded, no date (PFS 16.0 mo)",
    "P22": "recurrence recorded, no date (PFS 24.0 mo)",
    "P25": "recurrence recorded, no date (PFS 18.0 mo)",
    # P38 is NOT here on purpose. Only chordoma_sample_qc_full.csv claimed it
    # recurred; the survival sheet records Recurrence=N with no dates, and the
    # sheet is authoritative. Its PFS event at 43 months is the metastasis
    # (Metastasis=Y), not a recurrence. It classifies normally.
}


class PatientMethods:
    """
    Clinical metadata keyed by sample.

    Up to now every sample was assumed to be a different patient. That is not
    true: some patients contributed several samples, and one of them (P32)
    spans both sequencing runs — which is also why the cohort split stops being
    a useful grouping for the figures. Labels become p<patient>-s<sample>.

    On aggressiveness: chordoma has no standard aggressiveness grade. The
    literature models time-to-event (OS / chordoma-specific / recurrence-free
    survival), and the strongest reproducible factors are surgical margins,
    histologic subtype, Ki-67 and size — none of which are in this table. The
    default score here is therefore a construct, deliberately built from the
    two hard outcomes available (death, metastasis). Recurrence is displayed
    but not scored by default: in this cohort it is effectively binary (22
    patients at 0, 17 at exactly 1, one at 5), so weighting it would let
    "recurred once" outweigh mortality. Change `weights` to rescore.
    """

    DEFAULT_WEIGHTS = {"dead": 1.0, "metastasis": 1.0, "recurrence": 0.0}
    UNKNOWN_COLOR = "#c9c9c9"

    # Aggressiveness scheme presets. The weight schemes score terminal outcome;
    # "recurrence_timing" scores how fast the disease came back instead, and is
    # the only one of the three that can return "unknown" — see
    # recurrence_timing_class for why that matters.
    AGGRESSIVENESS_SCHEMES = {
        "outcome":         {"dead": 1.0, "metastasis": 1.0, "recurrence": 0.0},
        "outcome_plus_rec": {"dead": 1.0, "metastasis": 1.0, "recurrence": 1.0},
        "recurrence_timing": None,      # not weight-based; see apply_recurrence_timing
    }

    @staticmethod
    def _apply_survival_fixes(s: pd.DataFrame, id_col: str,
                              where: str, verbose: bool = True) -> pd.DataFrame:
        """
        Apply SURVIVAL_SHEET_FIXES to a freshly read survival sheet, loudly.

        Matches on `id_col`, which may carry several sequencing ids
        semicolon-joined for a patient who contributed more than one sample.
        """
        applied = []
        for sid, fix in SURVIVAL_SHEET_FIXES.items():
            hit = s[id_col].astype(str).apply(
                lambda cell: sid in [x.strip() for x in str(cell).split(";")])
            if not hit.any():
                continue
            for col, val in fix.items():
                if col not in s.columns:
                    continue
                for i in s.index[hit]:
                    old = s.at[i, col]
                    if pd.isna(old) or old != val:
                        s.at[i, col] = val
                        applied.append(f"{sid}.{col}: {old!r} -> {val!r}")
        if applied and verbose:
            print(f"[FIX] {where}: SURVIVAL_SHEET_FIXES applied — "
                  + "; ".join(applied))
        return s

    @staticmethod
    def recurrence_timing_class(
        meta: pd.DataFrame,
        surv_csv: str,
        early_months: float = 24.0,
        recurrence_time_limit: float = 60.0,
        many_recurrences: int = 3,
        binary: bool = True,
        use_pfs_fallback: bool = True,
        ignore_patients: "dict | list | None" = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Aggressiveness from how fast disease came back, not from how it ended.

        **Binary over everyone the sheet can actually time.**

            aggressive  first recurrence within `early_months` of diagnosis,
                        OR at least `many_recurrences` recurrences in total
            indolent    everything else
            unknown     ONLY the `ignore_patients` — held out, not labelled

        In this cohort that is 9 aggressive, 25 indolent, 6 unknown of 40.

        `binary=False` restores a three-state version that also requires
        `recurrence_time_limit` months of follow-up before calling a patient
        indolent, and returns "unknown" for anyone censored earlier or
        recurring in the gap between the two thresholds. That left 26 of 40
        patients unlabelled, which is why binary is the default.

        `recurrence_time_limit` is therefore read ONLY when binary=False; the
        binary path returns before any branch that uses it.

        What binary costs, so it is not a surprise later: 15 patients are
        called indolent on a median 31 months of follow-up rather than the 60
        the strict reading would want.

        **Recurrence is read from `surv_csv` only.** The QC table's
        n_recurrences column is not trustworthy — it disagrees with the sheet
        for P9, P18, P30, P38 and P40, and is the only source claiming P38
        recurred at all. `meta` is used here to map sequencing ids to patients
        and for nothing else.

        `ignore_patients` defaults to the module-level IGNORED_PATIENTS
        registry — currently the six patients who recur with no recurrence
        date recorded. They come back as "unknown" in both modes with the
        reason carried through, and the ignore list beats every other source
        including the PFS fallback. Pass {} to classify them anyway, or a
        dict/list of your own. Ignoring is scoped to THIS scheme: those patients
        still count in full under the outcome-weighted ones, where death and
        metastasis are recorded for all of them.

        Start of follow-up is Diagnosis_date, falling back to
        First_surgery_date. Returns one row per patient: cls, reason, n_rec,
        months_to_first_recurrence, followup_months.
        """
        if not os.path.exists(surv_csv):
            print(f"[WARN] survival sheet not found: {surv_csv}")
            return pd.DataFrame()
        if "patient" not in meta.columns:
            print("[WARN] meta needs a 'patient' column")
            return pd.DataFrame()

        s = pd.read_csv(surv_csv)
        id_col = "sample" if "sample" in s.columns else s.columns[0]
        s = PatientMethods._apply_survival_fixes(
            s, id_col, "survival sheet", verbose=verbose)
        raw2pat = meta["patient"].astype(str).to_dict()

        def _pat(cell):
            # one sheet row can carry every sequencing id a patient contributed,
            # semicolon-joined
            for rid in str(cell).split(";"):
                p = raw2pat.get(rid.strip())
                if p:
                    return p
            return None

        s["_pat"] = s[id_col].map(_pat)
        s = s[s["_pat"].notna()].drop_duplicates("_pat").set_index("_pat")

        def _d(col):
            return (pd.to_datetime(s[col], errors="coerce")
                    if col in s.columns else pd.Series(pd.NaT, index=s.index))

        start = _d("Diagnosis_date").fillna(_d("First_surgery_date"))
        first_rec = _d("First_recurrence_date")
        for c in ("Recurrence_1_date", "Recurrence_2_date",
                  "Recurrence_3_date", "Recurrence_4_date"):
            first_rec = first_rec.fillna(_d(c))
        last_fu = _d("Last_followup_date").fillna(_d("Death_date"))

        # PFS_months is the same interval measured another way, and where both
        # exist they agree EXACTLY — 0.0 months apart for all 15 patients with
        # a recurrence date, r = 1.0000 (those rows carry
        # PFS_basis="computed(dx/surgery)", i.e. derived from those dates).
        #
        # Seven patients have a recurrence with no date recorded but do have
        # PFS_months from the source file, with PFS_event=1. Reading only the
        # date columns throws their timing away and leaves them unknown, when
        # the interval — the only thing this classifier needs — is right
        # there. Used only as a FALLBACK: a real date always wins.
        pfs = pd.to_numeric(s.get("PFS_months"), errors="coerce") \
            if "PFS_months" in s.columns else pd.Series(np.nan, index=s.index)
        pfs_ev = pd.to_numeric(s.get("PFS_event(1=prog)"), errors="coerce") \
            if "PFS_event(1=prog)" in s.columns \
            else pd.Series(np.nan, index=s.index)

        # RECURRENCE COMES FROM THE SURVIVAL SHEET ONLY.
        # chordoma_sample_qc_full.csv carries an n_recurrences column that is
        # not reliable — it is the sole source claiming P38 recurred, and it
        # disagrees with the sheet for P9, P18, P30, P38 and P40. The sheet is
        # authoritative for recurrence, so `meta` is used here for the
        # patient mapping and nothing else.
        #
        # Within the sheet a DATE still outranks the count: P18 records
        # Recurrence=N with a count of 0 next to a dated recurrence, so the
        # effective count is the larger of the stated count and the number of
        # dates actually present.
        n_stated = pd.to_numeric(s.get("Number_of_recurrences"),
                                 errors="coerce").fillna(0.0)
        n_dated = sum(_d(c).notna().astype(int)
                      for c in ("Recurrence_1_date", "Recurrence_2_date",
                                "Recurrence_3_date", "Recurrence_4_date"))
        n_dated = n_dated.where(n_dated > 0,
                                _d("First_recurrence_date").notna().astype(int))
        n_rec = pd.concat([n_stated, n_dated.astype(float)], axis=1).max(axis=1)
        t_rec = (first_rec - start).dt.days / 30.4375
        t_fu = (last_fu - start).dt.days / 30.4375
        if "FollowUp_months" in s.columns:
            t_fu = t_fu.fillna(pd.to_numeric(s["FollowUp_months"],
                                             errors="coerce"))

        # Rule inside the sheet: a DATE is proof a recurrence happened and
        # fixes when, so it wins whenever it exists. Failing that, the count or
        # the Recurrence=Y flag means a recurrence happened at an unknown time,
        # which is "unknown", not "no recurrence". Absence of all three is
        # genuinely recurrence-free.
        sheet_n = n_stated
        flag_y = (s["Recurrence"].astype(str).str.strip().str.upper().eq("Y")
                  if "Recurrence" in s.columns
                  else pd.Series(False, index=s.index))
        if verbose:
            n_undated = int(((flag_y | (sheet_n > 0)) & t_rec.isna()).sum())
            print(f"  recurrence read from the survival sheet only "
                  f"(chordoma_sample_qc_full.csv ignored); "
                  f"{n_undated} patient(s) recur with no date recorded")

        # Evidence that a recurrence happened at all — sheet sources only.
        any_rec_v = ((n_rec > 0) | (sheet_n > 0) | flag_y | t_rec.notna())

        # PFS_event=1 is PROGRESSION, which is broader than recurrence — it can
        # be triggered by death or systemic progression in a patient who never
        # recurred. Gating the fallback on PFS_event alone pulled in P27, P34,
        # P36 and P37, all recurrence-free, and would have relabelled them
        # aggressive off a progression event. The fallback therefore requires
        # independent evidence of a recurrence, and only supplies its timing.
        # The ignore list wins over every other source, PFS included — that is
        # the point of listing a patient there.
        ign = IGNORED_PATIENTS if ignore_patients is None else ignore_patients
        ign = ({p: "ignored" for p in ign} if not isinstance(ign, dict)
               else dict(ign))
        ign = {p: r for p, r in ign.items() if p in s.index}
        if ign and verbose:
            print(f"  [INFO] {len(ign)} patient(s) on IGNORED_PATIENTS: "
                  f"{', '.join(map(str, sorted(ign)))} — "
                  + "held out of both arms (unknown)")

        not_ignored = pd.Series(~s.index.isin(list(ign)), index=s.index)
        from_pfs = ((t_rec.isna() & pfs.notna() & (pfs_ev == 1) & any_rec_v
                     & not_ignored)
                    if use_pfs_fallback
                    else pd.Series(False, index=s.index))
        if from_pfs.any() and verbose:
            print(f"  [INFO] recurrence timing taken from PFS_months for "
                  f"{int(from_pfs.sum())} patient(s) with a recurrence but no "
                  f"date: {', '.join(map(str, s.index[from_pfs]))}")

        rows = []
        for p in s.index:
            nr, tf = float(n_rec[p]), t_fu.get(p)
            tr = pfs.get(p) if bool(from_pfs.get(p, False)) else t_rec.get(p)
            _via = " (from PFS)" if bool(from_pfs.get(p, False)) else ""
            any_rec = bool(any_rec_v.get(p, False))
            n_eff = max(nr, float(sheet_n.get(p, 0.0)))
            if p in ign:
                # The ignore list is an exclusion, not a label: these patients
                # are held out of BOTH arms in both modes, because nothing in
                # the sheet can place their recurrence in time.
                cls, why = "unknown", f"ignored: {ign[p]}"
            elif n_eff >= many_recurrences:
                cls, why = "aggressive", f"{int(n_eff)} recurrences"
            elif pd.notna(tr) and tr <= early_months:
                cls, why = "aggressive", f"recurred at {tr:.1f} mo{_via}"
            elif binary:
                # Every patient who is NOT ignored and NOT aggressive is
                # indolent. Say WHY, so a row called indolent on thin evidence
                # is visible in the table rather than hidden behind a label.
                if pd.notna(tr):
                    cls, why = "indolent", f"recurred at {tr:.1f} mo{_via}"
                elif any_rec:
                    cls, why = "indolent", "recurred, date not recorded"
                elif pd.notna(tf):
                    cls, why = "indolent", f"no recurrence, {tf:.0f} mo follow-up"
                else:
                    cls, why = "indolent", "no recurrence, follow-up unrecorded"
            elif pd.notna(tr):
                if tr > recurrence_time_limit:
                    cls, why = "indolent", f"first recurrence at {tr:.0f} mo{_via}"
                else:
                    cls, why = "unknown", (
                        f"recurred at {tr:.1f} mo{_via}, between "
                        f"{early_months:g} and {recurrence_time_limit:g}")
            elif any_rec:
                cls, why = "unknown", "recurred, date not recorded"
            elif pd.notna(tf) and tf >= recurrence_time_limit:
                cls, why = "indolent", f"no recurrence, {tf:.0f} mo follow-up"
            else:
                cls, why = "unknown", (
                    f"censored at {tf:.0f} mo" if pd.notna(tf)
                    else "no follow-up recorded")
            rows.append({"patient": p, "cls": cls, "reason": why,
                         "n_recurrence": nr,
                         "months_to_first_recurrence": tr,
                         "followup_months": tf})
        out = pd.DataFrame(rows).set_index("patient")
        if verbose:
            c = out["cls"].value_counts()
            print(f"Recurrence timing: aggressive = recurred ≤{early_months:g} mo "
                  f"or ≥{many_recurrences} recurrences; "
                  + ("indolent = everything else, except the ignored patients "
                     "who are held out"
                     if binary
                     else f"indolent = no recurrence within {recurrence_time_limit:g} mo"))
            print(f"  {int(c.get('aggressive', 0))} aggressive, "
                  f"{int(c.get('indolent', 0))} indolent, "
                  f"{int(c.get('unknown', 0))} unknown of {len(out)} patients")
            if int(c.get("unknown", 0)):
                for why, k in (out.loc[out.cls == "unknown", "reason"]
                               .str.split(" at ").str[0].value_counts().items()):
                    print(f"    unknown — {why}: {k}")
        return out

    @staticmethod
    def recurrence_from_survival(
        meta: pd.DataFrame,
        surv_csv: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        `meta` with n_recurrence replaced by the survival sheet's value.

        chordoma_sample_qc_full.csv's n_recurrences column is wrong for some
        patients — it disagrees with the sheet for P9, P18, P30, P38 and P40 —
        and it does not only affect the aggressiveness label: it feeds the
        "outcome_plus_rec" scheme, the recurrence counts printed beside the
        swimmer plots, and the oncoprint's Recurrences annotation strip. This
        redirects all of them at the source.

        Within the sheet a recurrence DATE outranks the stated count, so the
        value is max(Number_of_recurrences, number of dates present) — that is
        what rescues P18, which records Recurrence=N and a count of 0 next to
        a dated recurrence, without the QC table being involved.

        Every change is printed.
        """
        if not os.path.exists(surv_csv) or "patient" not in meta.columns:
            print("[WARN] cannot redirect n_recurrence; leaving meta unchanged")
            return meta
        s = pd.read_csv(surv_csv)
        id_col = "sample" if "sample" in s.columns else s.columns[0]
        s = PatientMethods._apply_survival_fixes(
            s, id_col, "survival sheet", verbose=verbose)
        raw2pat = meta["patient"].astype(str).to_dict()

        def _pat(cell):
            for rid in str(cell).split(";"):
                p = raw2pat.get(rid.strip())
                if p:
                    return p
            return None

        s["_pat"] = s[id_col].map(_pat)
        s = s[s["_pat"].notna()].drop_duplicates("_pat").set_index("_pat")

        def _d(col):
            return (pd.to_datetime(s[col], errors="coerce")
                    if col in s.columns else pd.Series(pd.NaT, index=s.index))

        n_stated = pd.to_numeric(s.get("Number_of_recurrences"),
                                 errors="coerce").fillna(0.0)
        n_dated = sum(_d(c).notna().astype(int)
                      for c in ("Recurrence_1_date", "Recurrence_2_date",
                                "Recurrence_3_date", "Recurrence_4_date"))
        n_dated = n_dated.where(n_dated > 0,
                                _d("First_recurrence_date").notna().astype(int))
        eff = pd.concat([n_stated, n_dated.astype(float)], axis=1).max(axis=1)

        m = meta.copy()
        new = m["patient"].map(eff)
        if verbose and "n_recurrence" in m.columns:
            ch = m.loc[m["n_recurrence"].fillna(-1) != new.fillna(-1),
                       ["patient", "n_recurrence"]].copy()
            ch["from_sheet"] = new.reindex(ch.index)
            ch = ch.drop_duplicates("patient")
            if len(ch):
                print(f"[FIX] n_recurrence redirected to the survival sheet "
                      f"for {len(ch)} patient(s): " + "; ".join(
                          f"{r.patient}: {r.n_recurrence} -> {r.from_sheet:g}"
                          for r in ch.itertuples()))
        m["n_recurrence"] = new.astype(float)
        return m

    @staticmethod
    def apply_recurrence_timing(
        meta: pd.DataFrame,
        surv_csv: str,
        verbose: bool = True,
        **kw,
    ) -> pd.DataFrame:
        """
        `meta` with aggressiveness/aggressive/known rewritten from recurrence
        timing (see recurrence_timing_class).

        Unclassifiable patients get known=False and aggressiveness=NaN, which
        is what compare_groups and the group figures already filter on — so
        they drop out of comparisons rather than being counted as indolent,
        which is what setting aggressive=False alone would do.
        """
        cls = PatientMethods.recurrence_timing_class(meta, surv_csv,
                                                     verbose=verbose, **kw)
        if cls.empty:
            return meta
        m = meta.copy()
        per = m["patient"].map(cls["cls"])
        m["aggressiveness"] = per.map({"aggressive": 1.0, "indolent": 0.0})
        m["known"] = per.isin(("aggressive", "indolent"))
        m["aggressive"] = m["aggressiveness"].fillna(0.0) > 0
        m["agg_reason"] = m["patient"].map(cls["reason"])
        if verbose:
            n = int((~m["known"]).sum())
            print(f"  -> {len(m) - n} of {len(m)} samples carry a usable label; "
                  f"{n} marked known=False and excluded from group comparisons")
        return m


    # ── loading ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_sheet_fixes(df: pd.DataFrame, where: str) -> pd.DataFrame:
        """Overwrite the known bad cells listed in SHEET_FIXES, loudly."""
        applied = []
        for sid, fix in SHEET_FIXES.items():
            if sid not in df.index:
                continue
            for col, val in fix.items():
                if col not in df.columns:
                    continue
                old = df.at[sid, col]
                if pd.isna(old) or old != val:
                    df.at[sid, col] = val
                    applied.append(f"{sid}.{col}: {old} -> {val}")
        if applied:
            print(f"[FIX] {where}: SHEET_FIXES applied — " + "; ".join(applied))
        return df

    @staticmethod
    def load_patient_meta(
        csv_path: str,
        samples: "list | None" = None,
        include_unmapped: bool = True,
        weights: "dict | None" = None,
    ) -> pd.DataFrame:
        """
        Read chordoma_sample_meta.csv (one row per sample) and derive labels.

        Parameters
        ----------
        samples          : the cohort's raw sample ids. Samples present here but
                           absent from the CSV are kept when `include_unmapped`,
                           with every clinical field NaN and `known=False` — they
                           draw grey in the annotation strips. Set False to drop
                           them from the analysis entirely.
        weights          : {"dead":, "metastasis":, "recurrence":} — see the
                           class docstring for why recurrence defaults to 0.

        Returns a frame indexed by sample_id with: patient, sample_index, label
        ("p32-s1"), sex, n_recurrence, metastasis, dead, vital_status,
        aggressiveness, aggressive (bool), known (bool).
        """
        w = dict(PatientMethods.DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)

        if not os.path.exists(csv_path):
            print(f"[WARN] Patient metadata not found: {csv_path}")
            return pd.DataFrame()

        m = pd.read_csv(csv_path)

        # Two schemas are in use: the workbook expansion (sample_id, n_recurrence,
        # sample_index) and the NanoPlot QC table (sample, n_recurrences, run,
        # idx). Normalise to one set of names rather than maintaining two loaders.
        m = m.rename(columns={"sample": "sample_id", "n_recurrences": "n_recurrence"})
        if "sample_id" not in m.columns or "patient" not in m.columns:
            print("[WARN] metadata needs a sample/sample_id and a patient column")
            return pd.DataFrame()

        # "unassigned" in the QC table means no patient could be assigned — that
        # is the same state as absent, not a patient called "unassigned"
        unassigned = m["patient"].astype(str).str.lower().isin(
            ("unassigned", "unknown", "na", "nan", ""))
        if unassigned.any():
            print(f"[INFO] {int(unassigned.sum())} sample(s) marked unassigned "
                  f"in the metadata: {', '.join(m.loc[unassigned, 'sample_id'])}")
        m = m.loc[~unassigned].copy()

        for col in ("metastasis", "dead"):
            if col in m.columns:
                m[col] = (m[col].astype(str).str.strip().str.lower()
                          .isin(("true", "1", "y", "yes")))
        if "dead" not in m.columns and "vital_status" in m.columns:
            m["dead"] = m["vital_status"].astype(str).str.strip().str.lower().eq("dead")

        # sample_index: (run, idx) order within a patient when the QC table
        # supplies them — the acquisition order, as far as the ids record it.
        # Deliberately NOT called a timepoint: whether a patient's samples are
        # serial or multi-region is not recorded here.
        if {"run", "idx"}.issubset(m.columns):
            m = m.sort_values(["patient", "run", "idx"])
            m["sample_index"] = m.groupby("patient").cumcount() + 1
        elif "sample_index" not in m.columns:
            m = m.sort_values(["patient", "sample_id"])
            m["sample_index"] = m.groupby("patient").cumcount() + 1

        m["known"] = True
        m = m.set_index("sample_id")
        m = PatientMethods._apply_sheet_fixes(m, "patient metadata")

        if samples is not None:
            unmapped = [s for s in samples if s not in m.index]
            if unmapped:
                print(f"[INFO] {len(unmapped)} sample(s) not in the metadata: "
                      f"{', '.join(unmapped)}")
                if include_unmapped:
                    extra = pd.DataFrame(index=unmapped, columns=m.columns)
                    extra["known"] = False
                    extra["patient"] = [f"U{i+1}" for i in range(len(unmapped))]
                    extra["sample_index"] = 1
                    m = pd.concat([m, extra])
                    print("       kept as unknown (grey); "
                          "include_unmapped=False drops them instead.")
                else:
                    print("       dropped (include_unmapped=False).")
            m = m.reindex([s for s in samples if s in m.index])

        # p<patient>-s<sample>; the -s suffix only where a patient has >1 sample
        n_per = m.groupby("patient")["sample_index"].transform("size")
        m["label"] = np.where(
            n_per > 1,
            m["patient"].astype(str).str.lower() + "-s"
            + m["sample_index"].astype("Int64").astype(str),
            m["patient"].astype(str).str.lower())

        def _flag(col):
            s = m[col] if col in m.columns else pd.Series(False, index=m.index)
            return s.astype("boolean").fillna(False).astype(float)

        dead = _flag("dead")
        mets = _flag("metastasis")
        rec = pd.to_numeric(m.get("n_recurrence"), errors="coerce").fillna(0.0)
        m["aggressiveness"] = (w["dead"] * dead + w["metastasis"] * mets
                               + w["recurrence"] * rec)
        m.loc[~m["known"].astype(bool), "aggressiveness"] = np.nan
        m["aggressive"] = m["aggressiveness"] > 0

        known = m["known"].astype(bool)
        print(f"Metadata: {int(known.sum())} sample(s) across "
              f"{m.loc[known, 'patient'].nunique()} patient(s)"
              + (f", {int((~known).sum())} unknown" if (~known).any() else ""))
        print(f"  weights {w} -> aggressiveness "
              f"{m['aggressiveness'].min():.0f}–{m['aggressiveness'].max():.0f}; "
              f"{int(m['aggressive'].sum())} aggressive / "
              f"{int((~m['aggressive'] & known).sum())} indolent")
        return m

    # ── ordering ─────────────────────────────────────────────────────────────

    @staticmethod
    def order_samples(
        samples: list,
        meta: pd.DataFrame,
        has_methylation: "set | list | None" = None,
        keep_patients_together: bool = True,
    ) -> list:
        """
        Most aggressive first; within an aggressiveness tier, samples WITH
        methylation to the left; then patient number, then sample index.

        `keep_patients_together` (default) ranks on PATIENT-level values — a
        patient's tier is its own score, and it counts as having methylation if
        any of its samples does — so a multi-sample patient stays one adjacent
        block. Set False to rank strictly per sample, which sorts P32's single
        methylated sample away from its other four.

        Unknown-metadata samples sort last. They are not indolent, they are
        unmeasured; putting them in the indolent block would overstate its size.
        """
        methyl = set(has_methylation or ())

        def pnum(patient):
            try:
                return int(re.sub(r"\D", "", str(patient)) or 0)
            except ValueError:
                return 0

        if keep_patients_together and not meta.empty:
            in_meta = [s for s in samples if s in meta.index]
            by_patient = meta.loc[in_meta].groupby("patient")
            p_agg = by_patient["aggressiveness"].max()
            p_known = by_patient["known"].any()
            p_methyl = {p: any(s in methyl for s in g.index)
                        for p, g in by_patient}
        else:
            p_agg = p_known = None

        def key(s):
            if s not in meta.index:
                return (1, 0.0, 1, 10 ** 6, 0, str(s))
            r = meta.loc[s]
            patient = r.get("patient", "")
            known = bool(r.get("known", True))
            if p_agg is not None:
                agg = p_agg.get(patient)
                known = bool(p_known.get(patient, known))
                has_m = p_methyl.get(patient, False)
            else:
                agg = r.get("aggressiveness")
                has_m = s in methyl
            agg = -float(agg) if pd.notna(agg) else 0.0
            return (0 if known else 1,           # unknowns last
                    agg,                          # most aggressive first
                    0 if has_m else 1,            # methylation to the left
                    pnum(patient),                # numeric, so p2 precedes p16
                    int(r.get("sample_index") or 1),
                    str(s))

        return sorted(samples, key=key)

    @staticmethod
    def relabel(df: pd.DataFrame, meta: pd.DataFrame,
                col: str = "sample") -> pd.DataFrame:
        """Map raw sample ids onto p<N>-s<k> labels in any long table."""
        if df is None or df.empty or meta.empty:
            return df
        return df.assign(**{col: df[col].map(meta["label"]).fillna(df[col])})

    # ── aggressive vs indolent ───────────────────────────────────────────────

    @staticmethod
    def compare_groups(
        meta: pd.DataFrame,
        alt: "pd.DataFrame | None" = None,
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        beta: "pd.DataFrame | None" = None,
        genes: "list | None" = None,
        title: str = "Aggressive vs indolent",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> dict:
        """
        How aggressive cases differ from indolent ones across SV, CNV, SNV and
        methylation.

        Burden panels are drawn at SAMPLE level (every sample is a point) but
        tested at PATIENT level: P32 and P33 contribute 5 and 4 samples, and
        samples from one patient are not independent, so a sample-level test
        would be anti-conservative. Patient values are the mean across that
        patient's samples; the per-gene table uses the union of a patient's
        samples (a gene counts as altered if altered in any of them).

        Unknown-metadata samples are excluded from both groups — they are
        unmeasured, not indolent.

        Returns {"burden": per-sample frame, "patient": per-patient frame,
                 "genes": per-gene Fisher table, "stats": burden test table}.
        """
        from scipy import stats as _st

        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if meta is None or meta.empty:
            print("[WARN] No metadata — cannot split into groups.")
            return {}

        known = meta.loc[meta["known"].astype(bool)]
        if known.empty:
            print("[WARN] No samples with metadata.")
            return {}

        # ── per-sample burden ───────────────────────────────────────────────
        b = pd.DataFrame(index=known.index)
        b["patient"] = known["patient"]
        b["group"] = np.where(known["aggressive"], "aggressive", "indolent")

        if sv_all is not None and not sv_all.empty:
            b["SV"] = sv_all.groupby("sample").size().reindex(b.index)
        if snv_all is not None and not snv_all.empty:
            b["SNV"] = snv_all.groupby("sample").size().reindex(b.index)
        if cnv_all is not None and not cnv_all.empty and "cn" in cnv_all.columns:
            # fraction of called segment length that is not at the sample's own
            # baseline — a ploidy-robust "how rearranged is this genome"
            c = cnv_all.assign(_len=(cnv_all["end"] - cnv_all["start"]).clip(lower=1))
            gain, loss = CohortMethods._cnv_thresholds(
                c, "cn", 2.5, 1.5, "ploidy", verbose=False)
            c["_alt"] = [(cn >= gain.get(s, 2.5)) or (cn <= loss.get(s, 1.5))
                         for s, cn in zip(c["sample"], c["cn"])]
            frac = (c.assign(_w=c["_len"] * c["_alt"]).groupby("sample")["_w"].sum()
                    / c.groupby("sample")["_len"].sum())
            b["CNV"] = (frac * 100).reindex(b.index)
        if beta is not None and not beta.empty:
            b["Methylation"] = (beta.mean() * 100).reindex(b.index)

        metrics = [m for m in ("SV", "CNV", "SNV", "Methylation") if m in b.columns]
        if not metrics:
            print("[WARN] No burden tables supplied — nothing to compare.")
            return {}

        # ── collapse to patients for the tests ──────────────────────────────
        pat = b.groupby("patient")[metrics].mean()
        pat["group"] = b.groupby("patient")["group"].first()

        rows = []
        for m in metrics:
            a = pat.loc[pat["group"] == "aggressive", m].dropna()
            i = pat.loc[pat["group"] == "indolent", m].dropna()
            if len(a) < 2 or len(i) < 2:
                rows.append((m, len(a), len(i), np.nan, np.nan, np.nan))
                continue
            u, p = _st.mannwhitneyu(a, i, alternative="two-sided")
            rows.append((m, len(a), len(i), float(a.median()), float(i.median()), p))
        stats_tbl = pd.DataFrame(
            rows, columns=["metric", "n_aggressive", "n_indolent",
                           "median_aggressive", "median_indolent", "p_mannwhitney"])

        # ── per-gene Fisher, patient level ──────────────────────────────────
        gene_tbl = pd.DataFrame()
        if alt is not None and not alt.empty:
            a = alt[alt["sample"].isin(known.index)].copy()
            a["patient"] = a["sample"].map(known["patient"])
            a["group"] = a["sample"].map(
                dict(zip(known.index, np.where(known["aggressive"],
                                               "aggressive", "indolent"))))
            wide = (a.groupby(["gene", "patient"])["altered"].any().unstack()
                     .reindex(index=genes or sorted(a["gene"].unique())))
            grp = a.groupby("patient")["group"].first()
            grows = []
            for gene in wide.index:
                v = wide.loc[gene].dropna().astype(bool)
                g = grp.reindex(v.index)
                a1 = int((v & (g == "aggressive")).sum())
                a0 = int((~v & (g == "aggressive")).sum())
                i1 = int((v & (g == "indolent")).sum())
                i0 = int((~v & (g == "indolent")).sum())
                try:
                    _, p = _st.fisher_exact([[a1, a0], [i1, i0]])
                except ValueError:
                    p = np.nan
                grows.append((gene, a1, a1 + a0, i1, i1 + i0,
                              100 * a1 / max(a1 + a0, 1),
                              100 * i1 / max(i1 + i0, 1), p))
            gene_tbl = pd.DataFrame(
                grows, columns=["gene", "n_alt_aggressive", "n_aggressive",
                                "n_alt_indolent", "n_indolent",
                                "pct_aggressive", "pct_indolent", "p_fisher"])
            gene_tbl["q_bonferroni"] = (gene_tbl["p_fisher"]
                                        * len(gene_tbl)).clip(upper=1.0)
            gene_tbl = gene_tbl.sort_values("p_fisher")

        # ── figure ──────────────────────────────────────────────────────────
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.2), squeeze=False)
        axes = axes[0]
        colors = {"aggressive": "#c1272d", "indolent": "#2c7fb8"}
        labels = {"SV": "large SV calls", "CNV": "% genome altered (CN)",
                  "SNV": "PASS SNV/indel calls", "Methylation": "mean beta (%)"}

        for ax, m in zip(axes, metrics):
            for j, grp_name in enumerate(("aggressive", "indolent")):
                vals = b.loc[b["group"] == grp_name, m].dropna()
                if vals.empty:
                    continue
                # violinplot needs at least two distinct values for its KDE and
                # raises on a singular covariance otherwise, so a one-sample
                # group falls back to the points alone.
                if len(vals) >= 2 and vals.nunique() > 1:
                    parts = ax.violinplot([vals.values], positions=[j],
                                          widths=0.72, showextrema=False,
                                          showmedians=True)
                    for body in parts["bodies"]:
                        body.set_facecolor(colors[grp_name])
                        body.set_edgecolor(colors[grp_name])
                        body.set_alpha(0.25)
                        body.set_linewidth(1.2)
                    parts["cmedians"].set_color(colors[grp_name])
                    parts["cmedians"].set_linewidth(1.6)
                jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.28
                ax.scatter(j + jitter, vals, s=14, color=colors[grp_name],
                           alpha=0.75, linewidths=0, zorder=3)
            # The KDE tail runs past the data and, on counts and percentages,
            # past zero into values that cannot exist. Clip the view rather
            # than the density so the shape stays honest where it is real.
            col = b[m].dropna()
            if not col.empty:
                lo, hi = float(col.min()), float(col.max())
                pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi) * .1, 1.0)
                ax.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad,
                            hi + pad)
            row = stats_tbl.loc[stats_tbl["metric"] == m].iloc[0]
            p = row["p_mannwhitney"]
            ptxt = ("n too small" if pd.isna(p)
                    else f"p = {p:.3g}" + ("  *" if p < 0.05 else ""))
            ax.set_title(f"{m}\n{ptxt}", fontsize=_fs(9))
            ax.set_ylabel(labels.get(m, m), fontsize=_fs(8))
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"aggressive\n(n={int(row['n_aggressive'])} pts)",
                                f"indolent\n(n={int(row['n_indolent'])} pts)"],
                               fontsize=_fs(8))
            ax.tick_params(axis="y", labelsize=_fs(7))
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"{title} (Mann-Whitney per patient)",
                     fontsize=_fs(11), y=1.02)
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir,
                                 filename or "aggressive_vs_indolent.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        return {"burden": b, "patient": pat, "genes": gene_tbl,
                "stats": stats_tbl}


    # ── tumour purity ────────────────────────────────────────────────────────

    @staticmethod
    def plot_purity(
        cnv_all: pd.DataFrame,
        meta: "pd.DataFrame | None" = None,
        samples: "list | None" = None,
        sort_by_purity: bool = True,
        confidence_style: str = "violin",   # or "errorbar"
        show_confidence: bool = True,       # False -> purity + ploidy only, so
                                            # the two panels that matter are
                                            # twice as tall at poster size
        sample_labels: "dict | None" = None,
        label_rotation: int = 90,
        title: str = "Tumour purity and ploidy",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Per-sample tumour purity and ploidy, as estimated by Wakhan.

        Wakhan encodes its chosen solution in the folder name
        `<ploidy>_<purity>_<confidence>`, and load_wakhan_cn_segments parses that
        into `purity` / `ploidy` columns — so this needs no extra caller. Both
        are one value per sample, not per segment.

        Read the confidence with the estimate: a purity near 1.0 on a
        low-confidence solution usually means Wakhan could not separate tumour
        from normal, not that the sample is pure. Purity also bounds every CN
        call downstream — a 0.45-purity sample has its CN compressed toward 2,
        which is why the oncoprint uses ploidy-relative thresholds.

        Returns the per-sample table it plots.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if cnv_all is None or cnv_all.empty or "purity" not in cnv_all.columns:
            print("[INFO] No purity column — Wakhan segments are the source; "
                  "CNVkit and CORAL do not estimate purity.")
            return pd.DataFrame()

        # purity / ploidy / solution_score are one value per sample (they come
        # from the solution folder name), so `first` is right for those.
        # `confidence` is per SEGMENT, so it is summarised length-weighted —
        # a 40 Mb segment should not count the same as a 40 kb one.
        cols = [c for c in ("purity", "ploidy", "solution_score")
                if c in cnv_all.columns]
        tab = cnv_all.groupby("sample")[cols].first()
        conf_by_sample = {}
        if "confidence" in cnv_all.columns:
            w = cnv_all.assign(_len=(cnv_all["end"] - cnv_all["start"]).clip(lower=1))
            w = w.dropna(subset=["confidence"])
            if not w.empty:
                agg = w.groupby("sample").apply(
                    lambda g: pd.Series({
                        "confidence": float(np.average(g["confidence"],
                                                       weights=g["_len"])),
                        "conf_q25": float(g["confidence"].quantile(0.25)),
                        "conf_q75": float(g["confidence"].quantile(0.75)),
                        "n_segments": float(len(g)),
                    }), include_groups=False)
                tab = tab.join(agg)
                conf_by_sample = {k: g["confidence"].to_numpy()
                                  for k, g in w.groupby("sample")}
        if samples:
            # reindex on the FULL requested list: a sample with no Wakhan
            # solution stays as a labelled gap. Silently dropping those is why
            # this figure said n=45 while the oncoprint said n=47 — the two
            # ONTWGS12 samples with no Wakhan folder simply vanished.
            tab = tab.reindex(samples)
        missing = [str(i) for i in tab.index[tab["purity"].isna()]]
        if missing:
            print(f"[INFO] {len(missing)} sample(s) have no Wakhan solution, "
                  f"drawn as gaps: {', '.join(missing)}")
        if int(tab["purity"].notna().sum()) == 0:
            print("[WARN] No usable purity values.")
            return tab
        if sort_by_purity:
            # na_position keeps the no-solution samples together at the end
            tab = tab.sort_values("purity", ascending=False, na_position="last")

        grp = None
        if meta is not None and not meta.empty and "aggressive" in meta.columns:
            grp = meta["aggressive"].reindex(tab.index)
        colors = ["#c1272d" if g is True else "#2c7fb8" if g is False else "#c9c9c9"
                  for g in (grp if grp is not None else [None] * len(tab))]

        has_ploidy = "ploidy" in tab.columns
        has_conf   = (show_confidence and "confidence" in tab.columns
                      and tab["confidence"].notna().any())
        n_panels   = 1 + int(has_ploidy) + int(has_conf)
        fig, axes = plt.subplots(
            n_panels, 1, sharex=True,
            figsize=(max(9.0, len(tab) * 0.30), 2.6 + 2.1 * n_panels),
            squeeze=False)
        axes = axes[:, 0]
        x = np.arange(len(tab))

        ax = axes[0]
        ax.bar(x, np.nan_to_num(tab["purity"].to_numpy()), width=0.78, color=colors)
        for xi, v in zip(x, tab["purity"].to_numpy()):
            if not np.isfinite(v):
                ax.text(xi, 0.03, "no Wakhan", ha="center", va="bottom",
                        fontsize=_fs(5), color="#999999", rotation=90)
        ax.axhline(float(tab["purity"].median()), color="#444444", lw=0.9, ls="--",
                   label=f"median {tab['purity'].median():.2f}")
        ax.set_ylabel("Wakhan purity", fontsize=_fs(9))
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        # `solution_score` (third field of the Wakhan folder name) stays in the
        # returned table but is deliberately not drawn: on a 0-1 axis next to
        # purity and segment confidence it reads as the same kind of quantity,
        # which it is not.
        if grp is None:      # otherwise this is folded into the group legend
            ax.legend(fontsize=_fs(7), frameon=False, loc="upper right")

        if has_ploidy:
            ax2 = axes[1]
            ax2.bar(x, tab["ploidy"].to_numpy(), width=0.78, color="#8c8c8c")
            ax2.axhline(2.0, color="#444444", lw=0.9, ls=":", label="diploid")
            ax2.set_ylabel("Wakhan ploidy", fontsize=_fs(9))
            ax2.legend(fontsize=_fs(7), frameon=False, loc="upper right")
            ax2.spines[["top", "right"]].set_visible(False)

        if has_conf:
            ax3 = axes[1 + int(has_ploidy)]
            if confidence_style == "violin" and conf_by_sample:
                # the full per-segment distribution — a sample can have a high
                # mean and still carry a tail of poorly-fitted segments, which
                # a point-and-whisker hides
                data, pos = [], []
                for i, sname in enumerate(tab.index):
                    v = conf_by_sample.get(sname)
                    if v is None or len(v) < 2:
                        continue
                    data.append(v)
                    pos.append(i)
                if data:
                    parts = ax3.violinplot(data, positions=pos, widths=0.85,
                                           showextrema=False, showmedians=True)
                    for b in parts["bodies"]:
                        b.set_facecolor("#7aa87a")
                        b.set_edgecolor("none")
                        b.set_alpha(0.75)
                    if "cmedians" in parts:
                        parts["cmedians"].set_color("#2f4f2f")
                        parts["cmedians"].set_linewidth(0.9)
                # length-weighted mean on top: what the CN calls actually rest on
                ax3.plot(x, tab["confidence"].to_numpy(), "o", ms=2.4,
                         color="#c1272d", label="length-weighted mean", zorder=5)
            else:
                lo = (tab["confidence"] - tab["conf_q25"]).clip(lower=0)
                hi = (tab["conf_q75"] - tab["confidence"]).clip(lower=0)
                ax3.errorbar(x, tab["confidence"].to_numpy(),
                             yerr=[lo.to_numpy(), hi.to_numpy()],
                             fmt="o", ms=3.5, lw=0.8, color="#4d7c4d",
                             ecolor="#a8c4a8", capsize=0)
            med = float(tab["confidence"].median())
            ax3.axhline(med, color="#444444", lw=0.9, ls="--",
                        label=f"median {med:.2f}")
            ax3.set_ylabel("segment\nconfidence", fontsize=_fs(9))
            if "n_segments" in tab.columns:
                ax3.text(0.005, 0.03,
                         f"{int(tab['n_segments'].sum()):,} segments, "
                         f"{int(tab['n_segments'].min())}-{int(tab['n_segments'].max())} "
                         f"per sample",
                         transform=ax3.transAxes, fontsize=_fs(6.5), color="#666666")
            ax3.set_ylim(0, 1.05)
            ax3.legend(fontsize=_fs(7), frameon=False, loc="lower right", ncol=2)
            ax3.spines[["top", "right"]].set_visible(False)

        axes[-1].set_xticks(x)
        ticklabels = ([sample_labels.get(t, t) for t in tab.index]
                      if sample_labels else list(tab.index))
        axes[-1].set_xticklabels(
            ticklabels, rotation=label_rotation, fontsize=_fs(6),
            ha="right" if label_rotation not in (0, 90) else "center")

        axes[-1].set_xlim(-0.6, len(tab) - 0.4)

        if grp is not None:
            handles = [mpatches.Patch(color="#c1272d", label="aggressive"),
                       mpatches.Patch(color="#2c7fb8", label="indolent")]
            # only claim a "no metadata" category when one is actually drawn
            if any(c == "#c9c9c9" for c in colors):
                handles.append(mpatches.Patch(color="#c9c9c9", label="no metadata"))
            handles += axes[0].get_legend_handles_labels()[0]
            axes[0].legend(handles=handles, fontsize=_fs(7), frameon=False,
                           loc="upper right", ncol=len(handles),
                           bbox_to_anchor=(1.0, 1.16), borderaxespad=0)

        n_ok = int(tab["purity"].notna().sum())
        fig.suptitle(f"{title}  (n={len(tab)}"
                     + (f"; {n_ok} with a Wakhan solution)" if n_ok != len(tab)
                        else ")"), fontsize=_fs(12), y=0.98)
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "purity_ploidy.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return tab

    # ── within-patient comparison ────────────────────────────────────────────

    @staticmethod
    def plot_patient_tracks(
        patient: str,
        meta: pd.DataFrame,
        cnv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        region: "str | tuple | None" = None,
        flank_bp: int = 2_000_000,
        sv_min_len: int = 10_000,
        cn_max: "float | None" = None,
        mask_unmappable: bool = True,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ):
        """
        One row per sample of a multi-sample patient: copy number as a step
        profile with SV breakpoints marked underneath, so differences between a
        patient's samples are read vertically.

        `region` None draws the whole genome; a gene name or (chrom, start, end)
        draws that locus ± flank_bp.

        The rows are ordered s1..sN by the metadata's sample_index, which is
        (run, idx) order. **That is acquisition order as recorded in the sample
        ids, not a verified timeline** — whether a patient's samples are serial
        or multi-region is not in the metadata, so differences between rows may
        be evolution or may be spatial heterogeneity. Do not label these as
        timepoints until that is confirmed.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        rows = meta.loc[meta["patient"] == patient]
        if rows.empty:
            print(f"[WARN] {patient}: not in the metadata.")
            return
        labels = list(rows.sort_values("sample_index")["label"]
                      if "label" in rows.columns else rows.index)
        if len(labels) < 2:
            print(f"[INFO] {patient}: only one sample — nothing to compare.")
            return

        if region is None:
            chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
            offsets, ticks, total = CohortMethods._genome_offsets(chroms)
            win = None
        else:
            chrom, gs, ge, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
            win = (chrom, gs, ge)

        n = len(labels)
        # One shared y scale, or the rows cannot be compared — which is the only
        # reason to stack them. A handful of very high-CN segments would push an
        # auto scale to 175 and flatten everything else, so clip at the 99.5th
        # percentile across this patient's samples and mark what was clipped.
        if cn_max is None and cnv_all is not None and not cnv_all.empty:
            vals = cnv_all.loc[cnv_all["sample"].isin(labels), "cn"].dropna()
            cn_max = float(np.percentile(vals, 99.5)) if len(vals) else 6.0
            cn_max = float(np.clip(np.ceil(cn_max), 4.0, 12.0))
        cn_max = cn_max or 6.0

        fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                                 figsize=(14, 1.5 * n + 1.2), squeeze=False)
        axes = axes[:, 0]

        for ax, lab in zip(axes, labels):
            if win is None:
                CohortMethods._draw_genome_axis(ax, chroms, offsets, total)
            # copy number
            if cnv_all is not None and not cnv_all.empty:
                c = cnv_all[cnv_all["sample"] == lab]
                if win is not None:
                    c = c[(c["chrom"].map(CohortMethods._norm_chrom) == win[0])
                          & (c["end"] > win[1]) & (c["start"] < win[2])]
                n_clip = 0
                for r in c.itertuples():
                    # subtract the unmappable blocks instead of dropping the
                    # whole segment: Wakhan segments are often arm-length, so
                    # an overlap test removed entire chromosomes
                    pieces = (mask_subtract(r.chrom, r.start, r.end)
                              if mask_unmappable else [(r.start, r.end)])
                    y = min(r.cn, cn_max)
                    if r.cn > cn_max:
                        n_clip += 1
                    off = (offsets.get(CohortMethods._norm_chrom(r.chrom), 0)
                           if win is None else 0)
                    for ps, pe in pieces:
                        ax.plot([off + ps, off + pe], [y, y],
                                color="#d95f02" if r.cn > cn_max else "#2c7fb8",
                                lw=1.6, solid_capstyle="butt")
                if n_clip:
                    ax.text(0.995, 0.9, f"{n_clip} segment(s) > {cn_max:g}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=_fs(6), color="#d95f02")
            # SV breakpoints
            if sv_all is not None and not sv_all.empty:
                s = sv_all[(sv_all["sample"] == lab)
                           & ((sv_all["svtype"] == "BND")
                              | (sv_all["svlen"].abs() >= sv_min_len))]
                for r in s.itertuples():
                    for ch, pos in ((r.chrom, r.start), (r.chrom2, r.end)):
                        cn = CohortMethods._norm_chrom(ch)
                        if win is not None and (cn != win[0]
                                                or not win[1] <= pos <= win[2]):
                            continue
                        x = offsets.get(cn, 0) + pos if win is None else pos
                        ax.axvline(x, color=SVTYPE_COLORS.get(r.svtype, "#888888"),
                                   lw=0.7, alpha=0.55, ymax=0.18, zorder=1)
            if win is not None:
                IGVmethods._mark_gene(ax, IGVmethods.gene_span(region), win[0])
            ax.set_ylabel(lab, fontsize=_fs(8), rotation=0, ha="right", va="center",
                          labelpad=6)
            ax.set_ylim(0, cn_max * 1.08)
            ax.tick_params(axis="y", labelsize=_fs(6))
            ax.spines[["top", "right"]].set_visible(False)
            if win is None:
                ax.set_xlim(0, total)

        if win is None:
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels(chroms, fontsize=_fs(7))
            axes[-1].set_xlabel("genome position", fontsize=_fs(9))
            sub = "genome-wide"
        else:
            axes[-1].set_xlim(win[1], win[2])
            axes[-1].set_xlabel(f"chr{win[0]} position (bp)", fontsize=_fs(9))
            sub = rlabel

        fig.suptitle(title or f"{patient} — copy number and SV breakpoints "
                              f"across its {n} samples ({sub})",
                     fontsize=_fs(12), y=0.995)
        plt.tight_layout()

        if actual_dir is not None:
            tag = "genome" if win is None else str(region).replace("/", "_")
            fpath = os.path.join(actual_dir,
                                 filename or f"tracks_{patient.lower()}_{tag}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()


    @staticmethod
    def plot_patient_cn_delta(
        patient: "str | None",
        meta: pd.DataFrame,
        cnv_all: pd.DataFrame,
        samples: "list | None" = None,
        region: "str | tuple | None" = None,
        flank_bp: int = 2_000_000,
        bin_bp: int = 1_000_000,
        reference: str = "first",
        mask_unmappable: bool = True,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Explicit copy-number CHANGE between a patient's samples.

        Stacked profiles show the states; this shows the difference. Segment
        boundaries differ between samples, so both are put on a shared bin grid
        (length-weighted mean CN per bin) before subtracting — comparing raw
        segment lists sample-to-sample would compare different intervals.

        samples   : explicit list of sample LABELS to compare, overriding the
                    per-patient lookup. Use it to difference a set that does not
                    correspond to one patient — e.g. every sample from one
                    sequencing run. `patient` is then only used for the title,
                    and may be None.

        reference : "first"  — every row is sample_i minus the first sample
                    "median" — every row minus the per-bin median across the
                    set, which is better when there is no natural baseline
                    A SAMPLE LABEL — every row minus that sample, which is then
                    dropped from the rows. This is the pseudo-normal mode.

        On using one sample as a "normal": these are all tumours, so the
        baseline carries its own CN changes and every difference is
        tumour-minus-tumour, not tumour-minus-germline. A gain in the output can
        equally be a loss in the reference. If the baseline is from a DIFFERENT
        patient, the difference also contains germline copy-number variation
        between two people, which is not a somatic event at all.

        A caveat that dominates the interpretation: Wakhan's purity/ploidy fit
        is degenerate, and a sample given a different ploidy solution will show
        a whole-genome offset here that is arithmetic, not biology. Read the
        flat genome-wide shifts as suspect and the focal, chromosome-limited
        differences as real.

        Returns the binned CN matrix (bins x samples) it was computed from.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if samples is not None:
            labels = list(samples)
        else:
            rows = meta.loc[meta["patient"] == patient]
            labels = list(rows.sort_values("sample_index")["label"]
                          if "label" in rows.columns else rows.index)
        have = set(cnv_all["sample"])
        missing = [s for s in labels if s not in have]
        if missing:
            print(f"[WARN] no CN table for: {', '.join(map(str, missing))}")
        labels = [s for s in labels if s in have]

        # A named reference must be in the set and must sort first, so the
        # "first" arithmetic below is the only code path that ever runs.
        ref_named = reference not in ("first", "median")
        if ref_named:
            if reference not in labels:
                print(f"[WARN] reference {reference!r} is not among the samples "
                      f"with CN: {', '.join(map(str, labels))}")
                return pd.DataFrame()
            labels = [reference] + [s for s in labels if s != reference]

        who = patient or "samples"
        if len(labels) < 2:
            print(f"[INFO] {who}: fewer than two samples with CN — nothing "
                  f"to difference.")
            return pd.DataFrame()

        if region is None:
            chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
            offsets, ticks, total = CohortMethods._genome_offsets(chroms)
            spans = [(c, 0, HG38_CHROM_SIZES[c]) for c in chroms]
            win = None
        else:
            chrom, gs, ge, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
            spans = [(chrom, gs, ge)]
            offsets = {chrom: 0}
            win = (chrom, gs, ge)
            bin_bp = min(bin_bp, max(1_000, (ge - gs) // 400))

        # ── shared grid ─────────────────────────────────────────────────────
        grid = []
        for c, lo, hi in spans:
            for b in range(int(lo // bin_bp), int(np.ceil(hi / bin_bp))):
                grid.append((c, b * bin_bp, (b + 1) * bin_bp))
        gi = pd.MultiIndex.from_tuples([(c, s) for c, s, _ in grid],
                                       names=["chrom", "start"])
        mat = pd.DataFrame(index=gi, columns=labels, dtype=float)

        cnv = cnv_all.assign(chr_norm=cnv_all["chrom"].map(CohortMethods._norm_chrom))
        for lab in labels:
            sub = cnv.loc[cnv["sample"] == lab]
            acc = {}
            for r in sub.itertuples():
                if win is not None and (r.chr_norm != win[0]
                                        or r.end < win[1] or r.start > win[2]):
                    continue
                b0 = int(max(r.start, 0) // bin_bp)
                b1 = int(min(r.end, HG38_CHROM_SIZES.get(r.chr_norm, r.end)) // bin_bp)
                for b in range(b0, b1 + 1):
                    lo, hi = b * bin_bp, (b + 1) * bin_bp
                    w = max(0, min(r.end, hi) - max(r.start, lo))
                    if w <= 0:
                        continue
                    k = (r.chr_norm, lo)
                    acc.setdefault(k, [0.0, 0.0])
                    acc[k][0] += r.cn * w
                    acc[k][1] += w
            for k, (num, den) in acc.items():
                if k in mat.index and den > 0:
                    if mask_unmappable and is_masked_region(k[0], k[1], k[1] + bin_bp):
                        continue        # leave the bin NaN, so it is not drawn
                    mat.loc[k, lab] = num / den

        # ref_named was moved to the front of `labels` above, so it and
        # "first" share this branch.
        base = (mat.median(axis=1) if reference == "median" else mat[labels[0]])
        delta = mat.sub(base, axis=0)
        show = labels if reference == "median" else labels[1:]
        base_name = "the median" if reference == "median" else labels[0]

        # ── figure ──────────────────────────────────────────────────────────
        n = len(show)
        fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                                 figsize=(14, 1.35 * n + 1.2), squeeze=False)
        axes = axes[:, 0]
        lim = float(np.nanpercentile(np.abs(delta[show].to_numpy()), 99.5) or 1)
        lim = float(np.clip(np.ceil(lim), 1.0, 6.0))

        for ax, lab in zip(axes, show):
            xs, ys = [], []
            for (c, st) in delta.index:
                xs.append((offsets.get(c, 0) + st) if win is None else st)
                ys.append(delta.loc[(c, st), lab])
            xs, ys = np.array(xs, float), np.array(ys, float)
            if win is None:
                CohortMethods._draw_genome_axis(ax, chroms, offsets, total)
            ax.axhline(0, color="#444444", lw=0.7)
            # Pass the NaNs through rather than filtering them out: dropping the
            # masked bins first made fill_between span the gap with a straight
            # polygon, so a blanked centromere still looked like a solid bar.
            ax.fill_between(xs, np.clip(ys, 0, None), step="mid",
                            color="#c1272d", alpha=0.75, linewidth=0)
            ax.fill_between(xs, np.clip(ys, None, 0), step="mid",
                            color="#2c7fb8", alpha=0.75, linewidth=0)
            if win is not None:
                IGVmethods._mark_gene(ax, IGVmethods.gene_span(region), win[0])
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{lab}\n− {base_name}",
                          fontsize=_fs(7), rotation=0, ha="right", va="center", labelpad=8)
            ax.tick_params(axis="y", labelsize=_fs(6))
            ax.spines[["top", "right"]].set_visible(False)
            med = np.nanmedian(ys)
            if abs(med) > 0.25:
                ax.text(0.995, 0.94, f"genome-wide offset {med:+.1f} "
                                     f"— check the ploidy fit",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=_fs(6.5), color="#8a6d1f", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#d8c98a", linewidth=0.5))

        if win is None:
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels(chroms, fontsize=_fs(7))
            axes[-1].set_xlabel("genome position", fontsize=_fs(9))
            axes[-1].set_xlim(0, total)
            sub_t = "genome-wide"
        else:
            axes[-1].set_xlim(win[1], win[2])
            axes[-1].set_xlabel(f"chr{win[0]} position (bp)", fontsize=_fs(9))
            sub_t = rlabel

        fig.suptitle(title or f"{who} — copy-number change vs {base_name}"
                              f"  ({sub_t}, {bin_bp//1000} kb bins)",
                     fontsize=_fs(12), y=0.995)
        plt.tight_layout()

        if actual_dir is not None:
            tag = "genome" if win is None else str(region).replace("/", "_")
            fpath = os.path.join(actual_dir,
                                 filename or
                                 f"cndelta_{str(who).lower()}_{tag}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return mat


    @staticmethod
    def plot_patient_meth_delta(
        patient: "str | None",
        meta: pd.DataFrame,
        beta: "pd.DataFrame | None" = None,
        bed_paths: "dict | None" = None,
        region: "str | tuple | None" = None,
        flank_bp: int = 2_000_000,
        bin_bp: int = 1_000_000,
        reference: str = "first",
        min_cov: int = 5,
        mask_unmappable: bool = True,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Methylation CHANGE between a patient's samples — the 5mC counterpart of
        plot_patient_cn_delta, with the same reference semantics.

        Two data sources, because the right one differs by scale:

          region=None   genome-wide, from the cached `beta` matrix (CpG islands
                        x samples, beta 0-1). One row per island is already the
                        natural resolution; islands are binned to `bin_bp` by
                        mean so the axis matches the CN figure.
          region=...    one locus, from the modkit pileups in `bed_paths`
                        ({label: bedMethyl path}), read per CpG through
                        _meth_region and binned. Island-level beta is far too
                        coarse to say anything about a single promoter.

        Deltas are in PERCENTAGE POINTS, not beta units, so +20 means twenty
        points more methylated than the reference — readable against the 0-100
        the browser figures use.

        reference : "first", "median", or a sample LABEL, which is then
                    subtracted from the others and dropped from the rows.

        Unlike copy number, this has no ploidy-fit degeneracy — but it does
        depend on coverage: a bin where either sample is thinly covered moves on
        a handful of reads. Bins failing `min_cov` are left NaN rather than
        drawn as zero change.

        Returns the binned methylation matrix (bins x samples, percent).
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        rows = meta.loc[meta["patient"] == patient] if patient else meta
        labels = list(rows.sort_values("sample_index")["label"]
                      if "sample_index" in rows.columns and "label" in rows.columns
                      else rows.index)

        if region is None:
            if beta is None or beta.empty:
                print("[WARN] genome-wide view needs the beta matrix — pass beta=")
                return pd.DataFrame()
            have = set(beta.columns)
        else:
            if not bed_paths:
                print("[WARN] locus view needs bed_paths={label: bedMethyl}")
                return pd.DataFrame()
            have = set(bed_paths)
        missing = [l for l in labels if l not in have]
        if missing:
            print(f"[WARN] no methylation for: {', '.join(map(str, missing))}")
        labels = [l for l in labels if l in have]

        ref_named = reference not in ("first", "median")
        if ref_named:
            if reference not in labels:
                print(f"[WARN] reference {reference!r} has no methylation data; "
                      f"available: {', '.join(map(str, labels))}")
                return pd.DataFrame()
            labels = [reference] + [l for l in labels if l != reference]

        who = patient or "samples"
        if len(labels) < 2:
            print(f"[INFO] {who}: fewer than two samples with methylation — "
                  f"nothing to difference.")
            return pd.DataFrame()

        # ── shared grid ─────────────────────────────────────────────────────
        if region is None:
            chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
            offsets, ticks, total = CohortMethods._genome_offsets(chroms)
            win, rlabel = None, "genome-wide"
            # beta rows are "chr1:10,469-11,240" style locus keys
            loc = beta.index.to_series().astype(str).str.extract(
                r"^(?:chr)?([^:]+):([\d,]+)")
            ok = loc[0].notna() & loc[1].notna()
            if not ok.any():
                print("[WARN] beta index is not locus-keyed — cannot place rows "
                      "on the genome axis")
                return pd.DataFrame()
            b = beta.loc[ok.to_numpy()]
            chrom_of = loc.loc[ok, 0].map(CohortMethods._norm_chrom).to_numpy()
            start_of = (loc.loc[ok, 1].str.replace(",", "", regex=False)
                        .astype(np.int64).to_numpy())
            binned = (b[labels] * 100.0).copy()
            binned["_c"] = chrom_of
            binned["_s"] = (start_of // bin_bp) * bin_bp
            mat = (binned.groupby(["_c", "_s"])[labels].mean())
            mat.index.names = ["chrom", "start"]
        else:
            chrom, gs, ge, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
            win = (chrom, gs, ge)
            offsets, ticks, total, chroms = {chrom: 0}, None, None, [chrom]
            bin_bp = min(bin_bp, max(200, (ge - gs) // 400))
            cols = {}
            for lab in labels:
                d = PatientMethods._meth_region(bed_paths[lab], chrom, gs, ge,
                                                min_coverage=min_cov)
                if d.empty:
                    print(f"[WARN] no CpGs for {lab} at {rlabel}")
                    cols[lab] = pd.Series(dtype=float)
                    continue
                g = (d.assign(_s=(d["pos"] // bin_bp) * bin_bp)
                       .groupby("_s")["freq"].mean())
                cols[lab] = g
            mat = pd.DataFrame(cols).sort_index()
            mat.index = pd.MultiIndex.from_arrays(
                [[chrom] * len(mat), mat.index], names=["chrom", "start"])
            labels = [l for l in labels if mat[l].notna().any()]
            if ref_named and reference not in labels:
                print(f"[WARN] reference {reference!r} has no CpGs at {rlabel}")
                return pd.DataFrame()

        if mask_unmappable:
            keep = [not is_masked_region(c, st, st + bin_bp) for c, st in mat.index]
            mat = mat.loc[keep]

        base = (mat.median(axis=1) if reference == "median" else mat[labels[0]])
        delta = mat.sub(base, axis=0)
        show = labels if reference == "median" else labels[1:]
        base_name = "the median" if reference == "median" else labels[0]

        # ── figure ──────────────────────────────────────────────────────────
        n = len(show)
        fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                                 figsize=(14, 1.35 * n + 1.2), squeeze=False)
        axes = axes[:, 0]
        lim = float(np.nanpercentile(np.abs(delta[show].to_numpy()), 99.5) or 10)
        lim = float(np.clip(np.ceil(lim / 5) * 5, 10.0, 100.0))

        for ax, lab in zip(axes, show):
            xs = np.array([(offsets.get(c, 0) + st) if win is None else st
                           for c, st in delta.index], float)
            ys = delta[lab].to_numpy(float)
            if win is None:
                CohortMethods._draw_genome_axis(ax, chroms, offsets, total)
            ax.axhline(0, color="#444444", lw=0.7)
            # hyper red / hypo blue, matching the CN figure's gain/loss reading
            ax.fill_between(xs, np.clip(ys, 0, None), step="mid",
                            color="#c1272d", alpha=0.75, linewidth=0)
            ax.fill_between(xs, np.clip(ys, None, 0), step="mid",
                            color="#2c7fb8", alpha=0.75, linewidth=0)
            if win is not None:
                IGVmethods._mark_gene(ax, IGVmethods.gene_span(region), win[0])
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{lab}\n− {base_name}", fontsize=_fs(7), rotation=0,
                          ha="right", va="center", labelpad=8)
            ax.tick_params(axis="y", labelsize=_fs(6))
            ax.spines[["top", "right"]].set_visible(False)
            cov = float(np.isfinite(ys).mean() * 100)
            if cov < 90:
                ax.text(0.995, 0.94, f"{cov:.0f}% of bins covered",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=_fs(6.5), color="#8a6d1f", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="#d8c98a", linewidth=0.5))

        if win is None:
            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels(chroms, fontsize=_fs(7))
            axes[-1].set_xlabel("genome position", fontsize=_fs(9))
            axes[-1].set_xlim(0, total)
        else:
            axes[-1].set_xlim(win[1], win[2])
            axes[-1].set_xlabel(f"chr{win[0]} position (bp)", fontsize=_fs(9))

        unit = "CpG islands" if win is None else "CpG sites"
        fig.suptitle(title or f"{who} — methylation change vs {base_name}"
                              f"  ({rlabel}, {unit}, "
                              f"{bin_bp//1000 or 1} kb bins, % points)",
                     fontsize=_fs(12), y=0.995)
        plt.tight_layout()

        if actual_dir is not None:
            tag = "genome" if win is None else str(region).replace("/", "_")
            fpath = os.path.join(actual_dir,
                                 filename or
                                 f"methdelta_{str(who).lower()}_{tag}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return mat


    @staticmethod
    def plot_methylation_overlay(
        bed_paths: dict,
        region="TBXT",
        flank_bp: int = 100_000,
        highlight: "str | None" = None,
        order: "list | None" = None,
        smooth_bp: int = 2_000,
        detail_bp: "int | None" = None,
        detail_style: str = "band",
        band_bin_bp: "int | None" = None,
        show_line: bool = True,
        smooth_sites: "int | None" = None,
        min_cov: int = 5,
        cpg_islands: "str | None" = None,
        highlight_color: str = "#e8820c",
        other_colors: "list | None" = None,
        log_y: bool = False,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Every sample's methylation line at one locus, on ONE set of axes.

        The browser figure draws per-CpG points plus a rolling mean, one panel
        per sample. That is the right way to judge a single sample and the wrong
        way to compare several: the points dominate the ink, and the panels are
        stacked, so the eye cannot hold two traces at once. Here the scatter is
        dropped and only the smoothed lines are kept, superimposed.

        `cpg_islands` is the UCSC cpgIslandExt path; when given, islands inside
        the window are shaded. Pass None to skip that band.

        `highlight` is drawn in `highlight_color` and on top; every other sample
        gets a shade from `other_colors`, light to dark, in `order` (or sorted)
        sequence. One saturated colour against a graded set reads as
        "reference vs the rest" without a legend lookup for the reference.

        Smoothing is a centred rolling mean over a fixed GENOMIC WIDTH,
        `smooth_bp`, not over a fixed number of CpGs. That makes the line
        independent of the window: the same locus reads the same at +/-100 kb
        and at +/-2 Mb, and `flank_bp` only decides how much of it you see.

        A site-count window cannot do that, because the bp it spans depends on
        local CpG density and on nothing else. The browser's 1000-site setting
        spans ~180 bp inside a dense island but ~92 kb once the flanks are
        included, so a 1.6 kb unmethylated promoter — order 2% of what is being
        averaged — comes back as 60% when its true level is 4%. The dip was not
        absent from the wide view, it was averaged away.

        `smooth_sites` restores the old behaviour when passed, for parity with
        the browser figures; `smooth_bp` is ignored then.

        `detail_bp` draws a second, finer trace under each bold one at low
        alpha. It exists because the two things wanted from a wide view pull
        against each other: at 2 kb over a 4 Mb span the line is mostly noise
        and four samples are unreadable, while widening the smoothing to clean
        that up is exactly what flattens a promoter. Drawing both keeps the
        trend legible without throwing the detail away — set `smooth_bp` wide
        for the bold line and `detail_bp` to the fine width.

        `detail_style` decides how that finer trace is drawn:

          "band"  (default) the RANGE of the fine trace within each
                  `smooth_bp` window, as a filled band. Its floor is the real
                  minimum, so a 1.6 kb unmethylated promoter still reaches 0
                  in a 4 Mb view, where the bold mean can only show ~3% of the
                  dip's depth. Far less ink than the line, and it answers
                  "how low does it actually go here" directly.
          "line"  the fine trace itself, thin and faint. Truthful but busy at
                  megabase scale, which is what the band exists to fix.
          "none"  bold line only.

        Why a band rather than a narrower bold line: a linear smoother of width
        W renders a dip of width d at about background - (background-dip)*d/W.
        For d=1.6 kb to survive, W must be near 1.6 kb, which over 4 Mb is the
        noise you were trying to remove. The mean and the extremes answer
        different questions, so the figure shows both rather than compromising
        on one width.

        `show_line=False` replaces the curve with, per band bin, its MEDIAN as
        a line inside a faint full-range band. At a megabase view a curve
        smoothed over a fixed 2 kb is genuinely noisy — that noise is the data,
        not an artifact — and four of them overlaid are unreadable. Dropping the
        line loses no accuracy, because the band's edges ARE values of that same
        curve: the floor is its minimum in each bin and the ceiling its maximum.
        What is lost is central tendency, and the only honest way to restore it
        would be an average over the bin, which is the regional average this
        whole arrangement exists to avoid.

        `band_bin_bp` is how wide a slice the band summarises, and it is the
        ONLY setting here that may reasonably differ between views. It is a
        drawing choice: the band is always the min and max of the same
        `detail_bp` trace, so widening it draws fewer, wider boxes over the
        identical underlying values and never moves the line. Keep `smooth_bp`
        and `detail_bp` fixed across views so a locus reports the same number
        at every zoom, and let this absorb the density instead. Defaults to
        `smooth_bp`.

        Unlike the delta figures this shows absolute levels, so a band that is
        low in every sample is visible as such rather than cancelling to zero.

        `log_y` puts the percentage axis on a log scale, which spreads out the
        bottom of the range: on a linear axis 2% and 8% sit almost on top of
        each other near the floor, while being a fourfold difference in how
        methylated the locus is. It is the right view for reading depth of a
        promoter dip and the wrong one for comparing the 60-90% background,
        which it squashes. Note that a 0% bin cannot be drawn on a log axis at
        all, so the floor is clamped just under the smallest positive value.

        Returns one row per sample: n CpGs, mean and median percent methylation
        in the window.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if not bed_paths:
            print("[WARN] no bedMethyl paths supplied")
            return pd.DataFrame()

        chrom, gs, ge, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)

        def _smooth(d, width=None):
            """Centred mean over `width` bp of sequence, or over N sites."""
            if smooth_sites is not None:
                return d["freq"].rolling(smooth_sites, center=True,
                                         min_periods=1).mean().to_numpy()
            width = smooth_bp if width is None else width
            # bp window on irregularly spaced sites: prefix sums over the
            # half-open [pos-h, pos+h] slice each site sees. O(n), and exact —
            # a site-count rolling() cannot express this.
            pos = d["pos"].to_numpy()
            val = d["freq"].to_numpy(float)
            h = max(1, int(width) // 2)
            csum = np.concatenate([[0.0], np.cumsum(val)])
            lo = np.searchsorted(pos, pos - h, side="left")
            hi = np.searchsorted(pos, pos + h, side="right")
            n = np.maximum(hi - lo, 1)
            return (csum[hi] - csum[lo]) / n

        labels = list(order) if order else sorted(bed_paths)
        labels = [l for l in labels if l in bed_paths]
        if highlight and highlight in labels:
            labels = [l for l in labels if l != highlight] + [highlight]
        elif highlight:
            print(f"[WARN] highlight {highlight!r} not among the samples")
            highlight = None

        others = [l for l in labels if l != highlight]
        if other_colors is None:
            # light -> dark blue; a single "other" should not get the palest one
            ramp = ["#9ecae1", "#4292c6", "#2171b5", "#08519c", "#08306b"]
            other_colors = ([ramp[1]] if len(others) == 1 else
                            [ramp[int(round(k * (len(ramp) - 1)
                                            / max(len(others) - 1, 1)))]
                             for k in range(len(others))])
        col_of = {l: other_colors[k % len(other_colors)]
                  for k, l in enumerate(others)}
        if highlight:
            col_of[highlight] = highlight_color

        fig, ax = plt.subplots(figsize=(13, 4.6))
        rows = []
        for lab in labels:
            d = PatientMethods._meth_region(bed_paths[lab], chrom, gs, ge,
                                            min_coverage=min_cov)
            if d.empty:
                print(f"[WARN] no CpGs for {lab} at {rlabel}")
                continue
            d = d.sort_values("pos")
            line = _smooth(d)
            is_hi = (lab == highlight)
            band_med = None
            # fine detail first, so the bold trend sits on top of it
            if detail_bp and smooth_sites is None and detail_style != "none":
                fine = _smooth(d, detail_bp)
                if detail_style == "line":
                    ax.plot(d["pos"], fine, color=col_of[lab],
                            lw=0.5, alpha=0.30, zorder=(4 if is_hi else 2))
                else:
                    # range of the fine trace inside each bold-width window
                    pos_ = d["pos"].to_numpy()
                    _bw = int(band_bin_bp or smooth_bp)
                    key = (pos_ // max(1, _bw)).astype(np.int64)
                    g = pd.DataFrame({"k": key, "p": pos_, "v": fine}).groupby("k")
                    agg = g.agg(p=("p", "mean"), lo=("v", "min"),
                                hi=("v", "max"), med=("v", "median"))
                    ax.fill_between(agg["p"], agg["lo"], agg["hi"],
                                    color=col_of[lab],
                                    alpha=0.13 if not show_line else 0.16,
                                    linewidth=0, zorder=(4 if is_hi else 2))
                    band_med = agg
            if show_line:
                ax.plot(d["pos"], line, color=col_of[lab],
                        lw=2.4 if is_hi else 1.5, alpha=1.0 if is_hi else 0.9,
                        zorder=5 if is_hi else 3,
                        label=f"{lab} (reference)" if is_hi else lab)
            elif band_med is not None:
                # median of the curve within each band bin: one line per sample
                # that follows the band instead of filling it. It IS a
                # per-bin summary, so it does not carry the 2 kb curve's
                # extremes — the band around it still does, which is why both
                # are drawn.
                ax.plot(band_med["p"], band_med["med"], color=col_of[lab],
                        lw=2.4 if is_hi else 1.5,
                        alpha=1.0 if is_hi else 0.9, zorder=5 if is_hi else 3,
                        label=f"{lab} (reference)" if is_hi else lab)
            else:
                ax.plot([], [], color=col_of[lab], lw=6, alpha=0.45,
                        label=f"{lab} (reference)" if is_hi else lab)
            rows.append({"sample": lab, "n_cpg": len(d),
                         "mean_pct": float(d["freq"].mean()),
                         "median_pct": float(d["freq"].median())})

        if not rows:
            plt.close(fig)
            print(f"[WARN] nothing to draw at {rlabel}")
            return pd.DataFrame()

        IGVmethods._mark_gene(ax, IGVmethods.gene_span(region), chrom)
        if cpg_islands and os.path.exists(cpg_islands):
            isl = CohortMethods.load_regions_bed(cpg_islands)
            if not isl.empty and "chrom" in isl.columns:
                sub = isl[(isl["chrom"].astype(str).str.replace("chr", "",
                                                                regex=False)
                           == str(chrom))
                          & (isl["end"] >= gs) & (isl["start"] <= ge)]
                for r in sub.itertuples():
                    ax.axvspan(r.start, r.end, color="#2ca25f", alpha=0.10,
                               lw=0, zorder=0)

        ax.axhline(50, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(gs, ge)
        if log_y:
            ax.set_yscale("log")
            lo = min((r["median_pct"] for r in rows if r["median_pct"] > 0),
                     default=1.0)
            ax.set_ylim(max(0.1, lo / 20.0), 100)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(0, 100)
        # plain coordinates: matplotlib's offset notation turns a chr6 position
        # into "1.6615e8 +1e8", which is unreadable as a genome coordinate
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_ylabel("% methylation (rolling mean)"
                      + (", log scale" if log_y else ""), fontsize=_fs(9))
        ax.set_xlabel(f"chr{chrom} position (hg38)", fontsize=_fs(9))
        ax.tick_params(labelsize=_fs(7))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=_fs(7), frameon=False, ncol=min(4, len(rows)),
                  loc="lower center", bbox_to_anchor=(0.5, 1.01))
        _det = ("" if not detail_bp or detail_style == "none" else
                f", {detail_bp:,} bp "
                + ("range" if detail_style == "band" else "detail"))
        _sm = (f"{smooth_sites}-CpG" if smooth_sites is not None
               else (f"{smooth_bp:,} bp mean{_det}" if show_line
                     else f"median line and full range of the "
                          f"{detail_bp or smooth_bp:,} bp curve per "
                          f"{band_bin_bp or smooth_bp:,} bp"))
        _sm += " rolling mean" if smooth_sites is not None else ""
        ax.set_title(title or f"{rlabel} — methylation, {len(rows)} sample(s), "
                              f"{_sm}", fontsize=_fs(11), pad=30)
        plt.tight_layout()

        if actual_dir is not None:
            tag = str(region).replace("/", "_")
            fpath = os.path.join(actual_dir, filename or
                                 f"methoverlay_{tag}_{flank_bp//1000}kb"
                                 f"{'_log' if log_y else ''}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return pd.DataFrame(rows).set_index("sample")


    @staticmethod
    def plot_cn_overlay(
        cnv_all: pd.DataFrame,
        samples: list,
        region="TBXT",
        flank_bp: int = 100_000,
        highlight: "str | None" = None,
        order: "list | None" = None,
        highlight_color: str = "#e8820c",
        other_colors: "list | None" = None,
        log_y: bool = False,
        cn_max: "float | None" = None,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Copy-number traces at one locus, every sample on ONE set of axes — the
        CN counterpart of plot_methylation_overlay, with the same colouring:
        `highlight` in orange on top, the rest in a light-to-dark blue ramp.

        Drawn as STEPS, not lines. Copy number is piecewise constant between
        breakpoints, and interpolating across a segment boundary invents a ramp
        that the caller never reported — at this zoom a diagonal would be read
        as a gradual transition rather than a breakpoint.

        Segments are clipped to the window, so a segment running off the edge is
        drawn to the edge rather than being dropped.

        `log_y` uses a log2-style axis, on which equal ratios take equal
        vertical space: 1 -> 2 and 2 -> 4 are both one doubling, where a linear
        axis makes the second look twice as dramatic. CN 0 cannot be drawn on
        it, so homozygous deletions are clamped to the floor and the label says
        so — that is the case where linear is the honest view.

        Returns one row per sample: n segments in the window, and the
        length-weighted mean CN across it.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if cnv_all is None or cnv_all.empty:
            print("[WARN] no CN table supplied")
            return pd.DataFrame()

        chrom, gs, ge, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
        labels = list(order) if order else sorted(samples)
        labels = [l for l in labels if l in set(cnv_all["sample"])]
        missing = [l for l in (order or samples) if l not in labels]
        if missing:
            print(f"[WARN] no CN for: {', '.join(map(str, missing))}")
        if highlight and highlight in labels:
            labels = [l for l in labels if l != highlight] + [highlight]
        elif highlight:
            print(f"[WARN] highlight {highlight!r} has no CN in this table")
            highlight = None
        if not labels:
            print(f"[WARN] nothing to draw at {rlabel}")
            return pd.DataFrame()

        others = [l for l in labels if l != highlight]
        if other_colors is None:
            ramp = ["#9ecae1", "#4292c6", "#2171b5", "#08519c", "#08306b"]
            other_colors = ([ramp[1]] if len(others) == 1 else
                            [ramp[int(round(k * (len(ramp) - 1)
                                            / max(len(others) - 1, 1)))]
                             for k in range(len(others))])
        col_of = {l: other_colors[k % len(other_colors)]
                  for k, l in enumerate(others)}
        if highlight:
            col_of[highlight] = highlight_color

        cnv = cnv_all.assign(
            chr_norm=cnv_all["chrom"].map(CohortMethods._norm_chrom))
        floor = 0.05                      # log-axis floor for CN 0
        fig, ax = plt.subplots(figsize=(13, 4.6))
        rows, any_zero, top = [], False, 0.0
        for lab in labels:
            sub = cnv[(cnv["sample"] == lab) & (cnv["chr_norm"] == str(chrom))
                      & (cnv["end"] >= gs) & (cnv["start"] <= ge)]
            sub = sub.sort_values("start")
            if sub.empty:
                print(f"[WARN] no CN segments for {lab} at {rlabel}")
                continue
            xs, ys, wsum, wlen = [], [], 0.0, 0.0
            for r in sub.itertuples():
                a, b = max(int(r.start), gs), min(int(r.end), ge)
                if b <= a:
                    continue
                cn = float(r.cn)
                any_zero |= (cn <= 0)
                xs += [a, b]
                ys += [max(cn, floor) if log_y else cn] * 2
                wsum += cn * (b - a)
                wlen += (b - a)
            if not xs:
                continue
            is_hi = (lab == highlight)
            ax.plot(xs, ys, color=col_of[lab], lw=2.4 if is_hi else 1.5,
                    alpha=1.0 if is_hi else 0.9, zorder=5 if is_hi else 3,
                    drawstyle="steps-post",
                    label=f"{lab} (reference)" if is_hi else lab)
            top = max(top, max(ys))
            rows.append({"sample": lab, "n_segments": len(sub),
                         "mean_cn": wsum / wlen if wlen else np.nan})

        if not rows:
            plt.close(fig)
            return pd.DataFrame()

        IGVmethods._mark_gene(ax, IGVmethods.gene_span(region), chrom)
        ax.axhline(2, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(gs, ge)
        hi = cn_max if cn_max is not None else max(4.0, np.ceil(top) + 0.5)
        if log_y:
            ax.set_yscale("log", base=2)
            ax.set_ylim(floor * 0.8, hi)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(0, hi)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_ylabel("copy number" + (" (log2 axis)" if log_y else ""),
                      fontsize=_fs(9))
        ax.set_xlabel(f"chr{chrom} position (hg38)", fontsize=_fs(9))
        ax.tick_params(labelsize=_fs(7))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=_fs(7), frameon=False, ncol=min(4, len(rows)),
                  loc="lower center", bbox_to_anchor=(0.5, 1.01))
        _note = "  (CN 0 clamped to the floor)" if (log_y and any_zero) else ""
        ax.set_title(title or f"{rlabel} — copy number, {len(rows)} sample(s)"
                              f"{_note}", fontsize=_fs(11), pad=30)
        plt.tight_layout()

        if actual_dir is not None:
            tag = str(region).replace("/", "_")
            fpath = os.path.join(actual_dir, filename or
                                 f"cnoverlay_{tag}_{flank_bp//1000}kb"
                                 f"{'_log' if log_y else ''}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return pd.DataFrame(rows).set_index("sample")


    @staticmethod
    def plot_locus_group_compare(
        meta: pd.DataFrame,
        region="TBXT",
        flank_bp: int = 2_000_000,
        cnv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        sv_min_len: int = 10_000,
        germline_frac: float = 0.4,
        germline_af: float = 0.9,
        germline_slop: int = 1_000,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> dict:
        """
        Aggressive vs indolent at one locus, for copy number and for SV.

        CNV panels: per-sample copy number over the GENE BODY, shown both as an
        absolute value and as a multiple of that sample's own ploidy — these
        tumours span ploidy 1.6-5.5, so the absolute number on its own says more
        about whole-genome state than about the locus.

        SV panel: every breakend inside the window, drawn per sample and split
        by group, with the fraction of each group carrying one.

        Recurrent-event guard: a breakpoint seen in more than `germline_frac` of
        samples at allele fraction >= `germline_af` is flagged as a likely
        germline polymorphism. Positions are grouped on a `germline_slop` grid
        because callers place the same event a few bp apart in different samples and excluded from the group test. Without that,
        a common CNP near the gene dominates the comparison and produces a
        difference that is inherited, not somatic.

        Tests are per PATIENT (a patient's samples are not independent):
        Mann-Whitney for copy number, Fisher for SV presence.

        Returns {"cnv": per-sample CN table, "sv": windowed SV table,
                 "stats": test results, "germline": flagged breakpoints}.
        """
        from scipy import stats as _st

        actual_dir = SNVmethods._gene_save_dir(save_dir)
        chrom, ws, we, rlabel = IGVmethods.parse_region(region, flank_bp=flank_bp)
        span = IGVmethods.gene_span(region)
        gs, ge = (span[1], span[2]) if span else (ws, we)

        known = meta.loc[meta["known"].astype(bool)] if "known" in meta.columns else meta
        grp = pd.Series(np.where(known["aggressive"], "aggressive", "indolent"),
                        index=known.index)
        out = {"germline": pd.DataFrame(), "stats": pd.DataFrame()}

        # ── copy number over the gene body ──────────────────────────────────
        cn_tab = pd.DataFrame()
        if cnv_all is not None and not cnv_all.empty:
            c = cnv_all.assign(ch=cnv_all["chrom"].map(CohortMethods._norm_chrom))
            c = c[(c["ch"] == chrom) & (c["end"] > gs) & (c["start"] < ge)]
            if not c.empty:
                c = c.assign(_ov=(np.minimum(c["end"], ge)
                                  - np.maximum(c["start"], gs)).clip(lower=1))
                cn_tab = c.groupby("sample").apply(
                    lambda g: pd.Series({
                        "cn": float(np.average(g["cn"], weights=g["_ov"])),
                        "ploidy": float(g["ploidy"].dropna().iloc[0])
                        if g["ploidy"].notna().any() else np.nan,
                    }), include_groups=False)
                cn_tab["cn_rel"] = cn_tab["cn"] / cn_tab["ploidy"]
                cn_tab["group"] = grp.reindex(cn_tab.index)
                cn_tab["patient"] = known["patient"].reindex(cn_tab.index)
                cn_tab = cn_tab.dropna(subset=["group"])

        # ── SV breakends inside the window ──────────────────────────────────
        sv_w = pd.DataFrame()
        if sv_all is not None and not sv_all.empty:
            s = sv_all[(sv_all["svtype"] == "BND")
                       | (sv_all["svlen"].abs() >= sv_min_len)].copy()
            s["ch1"] = s["chrom"].map(CohortMethods._norm_chrom)
            s["ch2"] = s["chrom2"].map(CohortMethods._norm_chrom)
            hit = (((s["ch1"] == chrom) & s["start"].between(ws, we))
                   | ((s["ch2"] == chrom) & s["end"].between(ws, we)))
            sv_w = s[hit & s["sample"].isin(grp.index)].copy()

            if not sv_w.empty:
                # Group breakpoints on a slop grid, not on exact coordinates.
                # The same germline deletion is reported at 167,591,375 /
                # ...394 / ...395 in different samples, and exact-key grouping
                # splits it into several rare-looking events that then pass the
                # germline filter and read as somatic.
                sv_w["_k1"] = (sv_w["start"] // germline_slop).astype(int)
                sv_w["_k2"] = (sv_w["end"] // germline_slop).astype(int)
                key = sv_w.groupby(["chrom", "_k1", "_k2", "svtype"]).agg(
                    n_samples=("sample", "nunique"),
                    start=("start", "min"), end=("end", "max"),
                    af=("af", "median") if "af" in sv_w.columns else ("start", "size"))
                flag = key[(key["n_samples"] >= germline_frac * len(grp))
                           & (key["af"] >= germline_af)]
                if not flag.empty:
                    out["germline"] = flag.reset_index()
                    keys = set(map(tuple, flag.reset_index()[
                        ["chrom", "_k1", "_k2", "svtype"]].to_numpy()))
                    sv_w["germline"] = [tuple(t) in keys for t in
                                        sv_w[["chrom", "_k1", "_k2",
                                              "svtype"]].to_numpy()]
                    print(f"[INFO] {len(flag)} recurrent breakpoint(s) in "
                          f"{int(flag['n_samples'].max())}/{len(grp)} samples at "
                          f"AF>={germline_af} flagged as likely germline and "
                          f"excluded from the test.")
                else:
                    sv_w["germline"] = False
                sv_w["group"] = grp.reindex(sv_w["sample"]).to_numpy()
                sv_w["patient"] = known["patient"].reindex(sv_w["sample"]).to_numpy()

        # ── tests, per patient ──────────────────────────────────────────────
        rows = []
        for col, name in (("cn", "copy number"), ("cn_rel", "CN / ploidy")):
            if cn_tab.empty or col not in cn_tab:
                continue
            pat = cn_tab.groupby("patient").agg({col: "mean", "group": "first"})
            a = pat.loc[pat["group"] == "aggressive", col].dropna()
            i = pat.loc[pat["group"] == "indolent", col].dropna()
            if len(a) > 1 and len(i) > 1:
                _, p = _st.mannwhitneyu(a, i, alternative="two-sided")
            else:
                p = np.nan
            rows.append((name, len(a), len(i), float(a.median()),
                         float(i.median()), p))

        if sv_all is not None and not sv_all.empty and sv_w.empty:
            pat_grp = known.groupby("patient").first()["aggressive"]
            rows.append(("somatic SV in window", int(pat_grp.sum()),
                         int((~pat_grp).sum()), 0.0, 0.0, np.nan))
        if not sv_w.empty and "germline" in sv_w.columns:
            som = sv_w[~sv_w["germline"]]
            carriers = set(som["patient"].dropna())
            pat_grp = known.groupby("patient").first()["aggressive"]
            a1 = sum(1 for p, ag in pat_grp.items() if ag and p in carriers)
            a0 = sum(1 for p, ag in pat_grp.items() if ag and p not in carriers)
            i1 = sum(1 for p, ag in pat_grp.items() if not ag and p in carriers)
            i0 = sum(1 for p, ag in pat_grp.items() if not ag and p not in carriers)
            try:
                _, p = _st.fisher_exact([[a1, a0], [i1, i0]])
            except ValueError:
                p = np.nan
            rows.append((f"somatic SV in window", a1 + a0, i1 + i0,
                         100 * a1 / max(a1 + a0, 1), 100 * i1 / max(i1 + i0, 1), p))
        out["stats"] = pd.DataFrame(rows, columns=[
            "metric", "n_aggressive", "n_indolent",
            "aggressive", "indolent", "p"])

        # ── figure ──────────────────────────────────────────────────────────
        # Draw the SV panel whenever an SV table was supplied, even when the
        # window is empty: "no SV in any sample" is a finding, and a silently
        # missing panel reads as an oversight and breaks comparability between
        # the narrow and wide versions of the same figure.
        want_sv = sv_all is not None and not sv_all.empty
        panels = int(not cn_tab.empty) * 2 + int(want_sv)
        if panels == 0:
            print(f"[WARN] no CN segments or SV calls at {rlabel} — nothing to plot.")
            return out
        fig = plt.figure(figsize=(4.0 * max(panels, 2), 4.6))
        gs_ = fig.add_gridspec(1, max(panels, 1), wspace=0.35)
        colors = {"aggressive": "#c1272d", "indolent": "#2c7fb8"}
        k = 0

        for col, lab in (("cn", "copy number at gene body"),
                         ("cn_rel", "CN / sample ploidy")):
            if cn_tab.empty or col not in cn_tab:
                continue
            ax = fig.add_subplot(gs_[0, k]); k += 1
            for j, g in enumerate(("aggressive", "indolent")):
                v = cn_tab.loc[cn_tab["group"] == g, col].dropna()
                if v.empty:
                    continue
                ax.boxplot([v], positions=[j], widths=0.55, showfliers=False,
                           patch_artist=True,
                           boxprops=dict(facecolor=colors[g], alpha=0.25,
                                         edgecolor=colors[g]),
                           medianprops=dict(color=colors[g], lw=1.6),
                           whiskerprops=dict(color=colors[g]),
                           capprops=dict(color=colors[g]))
                jit = (np.random.RandomState(1).rand(len(v)) - 0.5) * 0.3
                ax.scatter(j + jit, v, s=16, color=colors[g], alpha=0.8,
                           linewidths=0, zorder=3)
            if col == "cn_rel":
                ax.axhline(1.0, color="#888888", lw=0.7, ls=":")
            r = out["stats"].loc[out["stats"]["metric"].str.startswith(
                "copy" if col == "cn" else "CN /")]
            pv = float(r["p"].iloc[0]) if len(r) else np.nan
            ax.set_title(f"{lab}\n" + ("n too small" if pd.isna(pv)
                                       else f"p = {pv:.3g}"), fontsize=_fs(9))
            ax.set_xticks([0, 1]); ax.set_xticklabels(["aggressive", "indolent"],
                                                      fontsize=_fs(8))
            ax.tick_params(axis="y", labelsize=_fs(7))
            ax.spines[["top", "right"]].set_visible(False)

        if want_sv:
            ax = fig.add_subplot(gs_[0, k])
            order = list(grp.index)
            ax.set_xlim(ws, we)
            ax.set_yticks([])
            ax.set_xlabel(f"chr{chrom} (bp)", fontsize=_fs(8))
            ax.tick_params(axis="x", labelsize=_fs(6))
            ax.spines[["top", "right"]].set_visible(False)
            IGVmethods._mark_gene(ax, span, chrom)

            if sv_w.empty:
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.5,
                        f"no SV \u2265{sv_min_len // 1000} kb\n"
                        f"in any of {len(order)} samples",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=_fs(9), color="#666666")
                ax.set_title("SV in window\n0 calls", fontsize=_fs(9))
            else:
                ypos = {sm: i for i, sm in enumerate(order)}
                for r in sv_w.itertuples():
                    y = ypos.get(r.sample)
                    if y is None:
                        continue
                    ax.plot([max(r.start, ws), min(r.end, we)], [y, y],
                            lw=2.0, alpha=0.85,
                            color=("#999999" if getattr(r, "germline", False)
                                   else colors.get(r.group, "#333333")))
                ax.set_ylim(-1, len(order))
                nsom = (int((~sv_w["germline"]).sum())
                        if "germline" in sv_w.columns else len(sv_w))
                ax.set_title(f"SV in window\n{nsom} somatic, "
                             f"{len(sv_w) - nsom} germline-flagged", fontsize=_fs(9))
                ax.legend(handles=[
                    mpatches.Patch(color=colors["aggressive"], label="aggressive"),
                    mpatches.Patch(color=colors["indolent"], label="indolent"),
                    mpatches.Patch(color="#999999", label="likely germline")],
                    fontsize=_fs(6.5), frameon=False, loc="upper right")

        fig.suptitle(title or f"{rlabel} — aggressive vs indolent "
                              f"(tests per patient)", fontsize=_fs(12), y=1.02)
        plt.tight_layout()

        if actual_dir is not None:
            tag = str(region).replace("/", "_")
            fpath = os.path.join(actual_dir, filename
                                 or f"locus_compare_{tag}_{flank_bp//1000}kb.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()

        out["cnv"], out["sv"] = cn_tab, sv_w
        return out


    @staticmethod
    def plot_arm_by_group(
        meta: pd.DataFrame,
        cnv_all: pd.DataFrame,
        chrom: str = "13",
        start: int = 20_000_000,
        end: int = 114_000_000,
        label: "str | None" = None,
        bin_bp: int = 1_000_000,
        mask_unmappable: bool = True,
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_baseline: str = "ploidy",
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> dict:
        """
        One chromosome arm, aggressive vs indolent: per-group loss/gain
        frequency, then every sample's copy-number profile as a heat row.

        Built for the chr13q result — 0/17 aggressive vs 6/21 indolent patients
        lose it — where the point is the *absence* of an event in one group.
        A frequency track alone cannot show absence convincingly; the per-sample
        rows can, because the reader sees there is nothing to miss.

        Copy number is scored against each sample's own ploidy by default, so
        rows are comparable across a cohort spanning ploidy 1.6-5.5.

        Test is per patient (Fisher on loss presence). Returns the per-sample
        calls and the test.
        """
        from scipy import stats as _st

        actual_dir = SNVmethods._gene_save_dir(save_dir)
        chrom = CohortMethods._norm_chrom(chrom)
        label = label or f"chr{chrom}:{start/1e6:.0f}-{end/1e6:.0f} Mb"

        known = meta.loc[meta["known"].astype(bool)] if "known" in meta else meta
        g_thr, l_thr = CohortMethods._cnv_thresholds(
            cnv_all, "cn", cnv_gain, cnv_loss, cnv_baseline, verbose=False)

        c = cnv_all.assign(ch=cnv_all["chrom"].map(CohortMethods._norm_chrom))
        c = c[(c["ch"] == chrom) & (c["end"] > start) & (c["start"] < end)]
        if c.empty:
            print(f"[WARN] no CN segments on {label}.")
            return {}

        n_bins = int(np.ceil((end - start) / bin_bp))
        samples = [s for s in known.index if s in set(c["sample"])]
        if not samples:
            print("[WARN] no samples with metadata and CN here.")
            return {}
        grp = pd.Series(np.where(known.loc[samples, "aggressive"],
                                 "aggressive", "indolent"), index=samples)

        prof = pd.DataFrame(np.nan, index=samples, columns=range(n_bins))
        for s in samples:
            for r in c[c["sample"] == s].itertuples():
                b0 = max(0, int((r.start - start) // bin_bp))
                b1 = min(n_bins - 1, int((r.end - start) // bin_bp))
                if b1 < b0:
                    continue
                base = g_thr.get(s, cnv_gain) / (cnv_gain / 2.0)
                val = r.cn / base if base else np.nan
                for b in range(b0, b1 + 1):
                    lo = start + b * bin_bp
                    if mask_unmappable and is_masked_region(chrom, lo, lo + bin_bp):
                        continue
                    prof.loc[s, b] = val

        # per-sample arm call, then per-patient
        call = {}
        for s in samples:
            v = prof.loc[s].dropna()
            if v.empty:
                call[s] = 0
                continue
            m = float(v.mean())
            call[s] = 1 if m >= cnv_gain / 2.0 else (-1 if m <= cnv_loss / 2.0 else 0)
        tab = pd.DataFrame({"group": grp, "call": pd.Series(call),
                            "patient": known.loc[samples, "patient"]})

        # Test BOTH directions. Reporting only loss made chr7 — a gain in a
        # quarter of the cohort — read as "0/17 vs 0/21 lose it, p = 1".
        rows_s = []
        for tgt, dname in ((-1, "loss"), (1, "gain")):
            pat = tab.assign(hit=tab["call"] == tgt).groupby("patient").agg(
                hit=("hit", "any"), group=("group", "first"))
            a1 = int(((pat["hit"]) & (pat["group"] == "aggressive")).sum())
            a0 = int(((~pat["hit"]) & (pat["group"] == "aggressive")).sum())
            i1 = int(((pat["hit"]) & (pat["group"] == "indolent")).sum())
            i0 = int(((~pat["hit"]) & (pat["group"] == "indolent")).sum())
            try:
                _, pv_ = _st.fisher_exact([[a1, a0], [i1, i0]])
            except ValueError:
                pv_ = np.nan
            rows_s.append({"region": label, "direction": dname,
                           "aggressive_hit": a1, "aggressive_n": a1 + a0,
                           "indolent_hit": i1, "indolent_n": i1 + i0,
                           "aggressive_pct": 100 * a1 / max(a1 + a0, 1),
                           "indolent_pct": 100 * i1 / max(i1 + i0, 1),
                           "p_fisher": pv_})
        stats_tbl = pd.DataFrame(rows_s)
        # headline = the direction that actually happens here
        lead = stats_tbl.iloc[int(np.argmax(
            stats_tbl["aggressive_hit"] + stats_tbl["indolent_hit"]))]
        a1, a0 = int(lead["aggressive_hit"]), int(lead["aggressive_n"] - lead["aggressive_hit"])
        i1, i0 = int(lead["indolent_hit"]), int(lead["indolent_n"] - lead["indolent_hit"])
        pv, verb = lead["p_fisher"], lead["direction"]

        # ── figure ──────────────────────────────────────────────────────────
        order = ([s for s in samples if grp[s] == "aggressive"]
                 + [s for s in samples if grp[s] == "indolent"])
        n_a = sum(grp[s] == "aggressive" for s in samples)
        fig, axes = plt.subplots(2, 1, sharex=True,
                                 figsize=(12, 1.2 + 0.20 * len(order) + 2.4),
                                 gridspec_kw={"height_ratios": [1.5, 4.0],
                                              "hspace": 0.08})
        xs = np.arange(n_bins) * bin_bp + start
        colors = {"aggressive": "#c1272d", "indolent": "#2c7fb8"}

        ax = axes[0]
        for gname in ("aggressive", "indolent"):
            gs_ = [s for s in samples if grp[s] == gname]
            sub = prof.loc[gs_]
            lost = (sub <= cnv_loss / 2.0).sum() / max(len(gs_), 1) * 100
            gained = (sub >= cnv_gain / 2.0).sum() / max(len(gs_), 1) * 100
            n_pat = known.loc[gs_, "patient"].nunique()
            ax.plot(xs, -lost.to_numpy(), color=colors[gname], lw=1.4,
                    label=f"{gname}: {len(gs_)} samples / {n_pat} patients")
            ax.plot(xs, gained.to_numpy(), color=colors[gname], lw=1.4, ls="--")
        ax.axhline(0, color="#444444", lw=0.7)
        ax.set_ylabel("% of group", fontsize=_fs(8))
        # Say what the two line styles are. The y-label alone ("loss down, gain
        # up") does not tell you that solid and dashed are different events.
        style = [mlines.Line2D([], [], color="#555555", lw=1.4,
                               label="loss (plotted downward)"),
                 mlines.Line2D([], [], color="#555555", lw=1.4, ls="--",
                               label="gain (plotted upward)")]
        grp_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=grp_handles + style, fontsize=_fs(7), frameon=False,
                  ncol=2, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{label} — {a1}/{a1+a0} aggressive vs {i1}/{i1+i0} "
                     f"indolent patients show {verb}   (Fisher p = "
                     + ("n/a" if pd.isna(pv) else f"{pv:.3g}") + ")", fontsize=_fs(10))

        ax = axes[1]
        im = ax.imshow(prof.loc[order].to_numpy(), aspect="auto",
                       cmap="RdBu_r", vmin=0, vmax=2,
                       extent=[start, end, len(order), 0], interpolation="nearest")
        # Shade the blanked columns. Without this a masked centromere and a
        # sample that simply has no segment there are both white, which are
        # very different statements.
        if mask_unmappable:
            for b in range(n_bins):
                lo = start + b * bin_bp
                if is_masked_region(chrom, lo, lo + bin_bp):
                    ax.axvspan(lo, lo + bin_bp, color="#d9d9d9", zorder=3,
                               linewidth=0)
            ax.text(0.002, -0.06, "grey = centromere/telomere/segdup (masked);  "
                                  "white = no segment called",
                    transform=ax.transAxes, fontsize=_fs(6), color="#666666")
        ax.axhline(n_a, color="black", lw=1.2)
        ax.set_yticks(np.arange(len(order)) + 0.5)
        ax.set_yticklabels(order, fontsize=_fs(5))
        for t, s in zip(ax.get_yticklabels(), order):
            t.set_color(colors[grp[s]])
        ax.set_xlabel(f"chr{chrom} position (bp)", fontsize=_fs(9))
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("CN / sample ploidy", fontsize=_fs(7))
        cb.ax.tick_params(labelsize=_fs(6))

        fig.suptitle(title or f"{label} by outcome group", fontsize=_fs(12), y=0.995)
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename
                                 or f"arm_{chrom}_by_group.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return {"calls": tab, "profile": prof, "stats": stats_tbl}


    @staticmethod
    def _raw_ids(meta: pd.DataFrame, idx: list) -> list:
        """
        Raw ONTWGS ids for plotting labels, whichever way `meta` is oriented.

        The metadata frame gets passed around both ways: indexed by raw id with
        a `label` column, and indexed by label with the raw id demoted to a
        column by reset_index(). Assuming the first is why the QC join silently
        produced a table of NaNs.
        """
        if "label" in meta.columns:                      # indexed by raw id
            return [{lab: raw for raw, lab in meta["label"].items()}.get(s, s)
                    for s in idx]
        for col in ("sample_id", "index", "level_0", "sample"):
            if col in meta.columns and meta[col].astype(str).str.startswith(
                    "ONTWGS").any():
                return [meta[col].get(s, s) for s in idx]
        return list(idx)

    @staticmethod
    def build_summary_table(
        meta: pd.DataFrame,
        sv_all: "pd.DataFrame | None" = None,
        snv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        qc_csv: "str | None" = None,
        samples: "list | None" = None,
        highlight_samples: "list | set | None" = None,
        cnv_source_of: "dict | None" = None,
        sv_min_len: int = 10_000,
        inv_types: "tuple" = ("INV", "INVDUP", "DEL/INV"),
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_baseline: str = "ploidy",
        save_dir: "str | None" = None,
        filename: "str | None" = "sample_summary.csv",
    ) -> pd.DataFrame:
        """
        One row per sample: clinical fields plus variant, copy-number and read
        counts — the numbers a methods table or a supplementary sheet needs.

        SV is reported twice on purpose. `n_sv` excludes inversions; `n_sv_inv`
        is the total including them. Sniffles inversion calls on ONT are the
        least trustworthy class (they arise readily from mapping around
        repeats), so a count that mixes them in is not comparable across
        samples with different repeat-region coverage. Same reasoning for
        splitting SNV from INDEL: ONT indel calls carry a different error
        profile from substitutions and should not be pooled into one "mutation
        burden".

        `qc_csv` is the NanoPlot table; its read-length and quality columns are
        joined on the raw sample id when supplied.

        Returns the table and writes it as CSV when `save_dir` is given.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if meta is None or meta.empty:
            print("[WARN] No metadata — cannot build the table.")
            return pd.DataFrame()

        idx = [s for s in (samples or meta.index) if s in meta.index]
        t = pd.DataFrame(index=pd.Index(idx, name="sample"))

        keep = [c for c in ("patient", "sample_index", "sex", "n_recurrence",
                            "metastasis", "vital_status", "dead", "aggressive",
                            "aggressiveness", "run", "idx") if c in meta.columns]
        for c in keep:
            t[c] = meta.loc[idx, c]

        # ── structural variants, with and without inversions ───────────────
        if sv_all is not None and not sv_all.empty:
            sv = sv_all[(sv_all["svtype"] == "BND")
                        | (sv_all["svlen"].abs() >= sv_min_len)]
            is_inv = sv["svtype"].isin(inv_types)
            t["n_sv"] = sv.loc[~is_inv].groupby("sample").size().reindex(idx).fillna(0).astype(int)
            t["n_sv_inv"] = sv.groupby("sample").size().reindex(idx).fillna(0).astype(int)
            for st in ("DEL", "DUP", "BND", "INV"):
                t[f"n_{st.lower()}"] = (sv[sv["svtype"] == st].groupby("sample")
                                        .size().reindex(idx).fillna(0).astype(int))

        # ── small variants, SNV and INDEL kept apart ───────────────────────
        if snv_all is not None and not snv_all.empty:
            if "vtype" in snv_all.columns:
                bt = (snv_all.groupby(["sample", "vtype"]).size().unstack(fill_value=0)
                      .reindex(idx).fillna(0))
                t["n_snv"] = bt["SNV"].astype(int) if "SNV" in bt else 0
                t["n_indel"] = bt["INDEL"].astype(int) if "INDEL" in bt else 0
            else:
                t["n_snv"] = snv_all.groupby("sample").size().reindex(idx).fillna(0).astype(int)

        # ── copy number ────────────────────────────────────────────────────
        if cnv_all is not None and not cnv_all.empty and "cn" in cnv_all.columns:
            c = cnv_all[cnv_all["sample"].isin(idx)].copy()
            if not c.empty:
                c["_len"] = (c["end"] - c["start"]).clip(lower=1)
                g_thr, l_thr = CohortMethods._cnv_thresholds(
                    c, "cn", cnv_gain, cnv_loss, cnv_baseline, verbose=False)
                c["_alt"] = [(cn >= g_thr.get(s, cnv_gain)) or (cn <= l_thr.get(s, cnv_loss))
                             for s, cn in zip(c["sample"], c["cn"])]
                t["n_cnv_segments"] = c.groupby("sample").size().reindex(idx)
                frac = (c.assign(_w=c["_len"] * c["_alt"]).groupby("sample")["_w"].sum()
                        / c.groupby("sample")["_len"].sum())
                t["pct_genome_cn_altered"] = (frac * 100).reindex(idx).round(1)
                for col in ("purity", "ploidy"):
                    if col in c.columns:
                        t[col] = c.groupby("sample")[col].first().reindex(idx)

        # ── NanoPlot read QC, joined on the raw id ─────────────────────────
        if qc_csv and os.path.exists(qc_csv):
            q = pd.read_csv(qc_csv).rename(columns={"sample": "sample_id"})
            q = q.set_index("sample_id")
            raw_idx = PatientMethods._raw_ids(meta, idx)
            for col in ("median_read_length", "n50", "mean_read_length",
                        "median_read_quality", "pct_above_q15", "n_reads",
                        "total_gb", "pct_identity", "frac_aligned",
                        "pass_qc", "qc_flags"):
                if col in q.columns:
                    t[col] = [q[col].get(r, np.nan) for r in raw_idx]
            t["sample_id"] = raw_idx
        else:
            t["sample_id"] = PatientMethods._raw_ids(meta, idx)

        if highlight_samples:
            t["clinical_flag"] = [s in set(highlight_samples) for s in idx]
        if cnv_source_of:
            # which caller each sample's CN numbers came from — the sources are
            # not interchangeable, so this belongs beside the numbers
            t["cnv_source"] = [cnv_source_of.get(s, "none") for s in idx]

        front = [c for c in ("patient", "sample_id") if c in t.columns]
        t = t[front + [c for c in t.columns if c not in front]]

        if actual_dir is not None and filename:
            fpath = os.path.join(actual_dir, filename)
            t.to_csv(fpath)
            print(f"Saved: {fpath}  ({t.shape[0]} rows x {t.shape[1]} columns)")
        return t


    @staticmethod
    def cnv_with_fallback(
        cnv_sources: dict,
        samples: "list | None" = None,
        priority: tuple = ("wakhan", "cnvkit", "coral_amplicon"),
    ) -> tuple:
        """
        One CN table per sample, taking the first source in `priority` that has
        segments for it, plus a {sample: source} map.

        The sources are NOT equivalent and the map exists so that is never
        forgotten:
          · wakhan          purity/ploidy-corrected, genome-wide
          · cnvkit          log2 ratio vs a flat diploid reference, genome-wide
          · coral_amplicon  segments only INSIDE amplified regions — not a
                            genome-wide profile, so anything computed from it
                            as a fraction of the genome is not comparable with
                            the other two.

        Returns (combined DataFrame with a `cnv_source` column, {sample: source}).
        """
        have = {k: v for k, v in (cnv_sources or {}).items()
                if v is not None and not v.empty}
        if not have:
            print("[WARN] No CNV sources supplied.")
            return pd.DataFrame(), {}

        wanted = list(samples) if samples is not None else sorted(
            {s for v in have.values() for s in v["sample"].unique()})

        frames, src_of, missing = [], {}, []
        for s in wanted:
            for src in priority:
                v = have.get(src)
                if v is None:
                    continue
                sub = v[v["sample"] == s]
                if sub.empty:
                    continue
                frames.append(sub.assign(cnv_source=src))
                src_of[s] = src
                break
            else:
                missing.append(s)

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        used = pd.Series(src_of).value_counts().to_dict() if src_of else {}
        print(f"[INFO] CNV source per sample: {used}"
              + (f"; no CNV at all for {len(missing)}: {', '.join(missing)}"
                 if missing else ""))
        non_primary = [s for s, src in src_of.items() if src != priority[0]]
        if non_primary:
            print(f"[INFO] fell back off {priority[0]} for: "
                  f"{', '.join(f'{s}({src_of[s]})' for s in non_primary)}")
        return out, src_of

    @staticmethod
    def plot_subset_burden(
        samples: list,
        meta: pd.DataFrame,
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        cnv_source_of: "dict | None" = None,
        sv_exclude_types: "tuple" = ("INV", "INVDUP", "DEL/INV"),
        sv_min_len: int = 10_000,
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_baseline: str = "ploidy",
        cohort_reference: bool = True,
        title: str = "Burden — selected samples",
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        SNV, SV and CNV burden for a chosen handful of samples, one panel each.

        `cohort_reference` draws the whole-cohort median as a dashed line behind
        each panel, so a selected sample is read against the cohort rather than
        only against the other five.

        SV excludes inversions (see build_summary_table for why); SNV and INDEL
        are stacked but kept distinguishable. CNV is percent of called segment
        length away from the sample's own ploidy.

        Returns the per-sample numbers it plots.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        sel = [s for s in samples if s in set(meta.index)]
        if not sel:
            print("[WARN] None of the requested samples are in the metadata.")
            return pd.DataFrame()

        def _snv(df):
            if df is None or df.empty:
                return None, None
            if "vtype" in df.columns:
                bt = df.groupby(["sample", "vtype"]).size().unstack(fill_value=0)
                return (bt["SNV"] if "SNV" in bt else None,
                        bt["INDEL"] if "INDEL" in bt else None)
            return df.groupby("sample").size(), None

        n_snv, n_indel = _snv(snv_all)

        n_sv = None
        if sv_all is not None and not sv_all.empty:
            sv = sv_all[(sv_all["svtype"] == "BND")
                        | (sv_all["svlen"].abs() >= sv_min_len)]
            sv = sv[~sv["svtype"].isin(sv_exclude_types)]
            n_sv = sv.groupby("sample").size()

        pct_cn = None
        if cnv_all is not None and not cnv_all.empty and "cn" in cnv_all.columns:
            c = cnv_all.assign(_len=(cnv_all["end"] - cnv_all["start"]).clip(lower=1))
            g_thr, l_thr = CohortMethods._cnv_thresholds(
                c, "cn", cnv_gain, cnv_loss, cnv_baseline, verbose=False)
            c["_alt"] = [(cn >= g_thr.get(s, cnv_gain)) or (cn <= l_thr.get(s, cnv_loss))
                         for s, cn in zip(c["sample"], c["cn"])]
            pct_cn = (c.assign(_w=c["_len"] * c["_alt"]).groupby("sample")["_w"].sum()
                      / c.groupby("sample")["_len"].sum() * 100)

        panels = [("SNV", n_snv, "ClairS-TO PASS SNV"),
                  ("SV", n_sv, f"SV \\u2265{sv_min_len//1000} kb (no INV)"),
                  ("CNV", pct_cn, "% of called CN away from ploidy")]
        panels = [p for p in panels if p[1] is not None]
        if not panels:
            print("[WARN] No burden tables supplied.")
            return pd.DataFrame()

        out = pd.DataFrame(index=pd.Index(sel, name="sample"))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.6),
                                 squeeze=False)
        axes = axes[0]
        agg = meta.get("aggressive")

        for ax, (key, series, ylab) in zip(axes, panels):
            vals = series.reindex(sel)
            out[key] = vals
            colors = ["#c1272d" if (agg is not None and bool(agg.get(s)))
                      else "#2c7fb8" for s in sel]
            xs = np.arange(len(sel))
            ax.bar(xs, np.nan_to_num(vals.to_numpy(float)), width=0.7, color=colors)
            if key == "SNV" and n_indel is not None:
                iv = n_indel.reindex(sel).fillna(0)
                out["INDEL"] = iv
                ax.bar(xs, iv.to_numpy(float),
                       bottom=np.nan_to_num(vals.to_numpy(float)),
                       width=0.7, color="#22405e", label="indel")
                ax.legend(fontsize=_fs(7), frameon=False)
            if cohort_reference:
                med = float(np.nanmedian(series.to_numpy(float)))
                ax.axhline(med, color="#444444", lw=0.9, ls="--",
                           label=f"cohort median {med:,.0f}")
                ax.legend(fontsize=_fs(7), frameon=False)
            for xi, v, s in zip(xs, vals.to_numpy(float), sel):
                if not np.isfinite(v):
                    ax.text(xi, 0, "no data", rotation=90, fontsize=_fs(6),
                            ha="center", va="bottom", color="#999999")
            ax.set_xticks(xs)
            ax.set_xticklabels(sel, rotation=45, ha="right", fontsize=_fs(8))
            ax.set_ylabel(ylab, fontsize=_fs(8))
            ax.tick_params(axis="y", labelsize=_fs(7))
            ax.spines[["top", "right"]].set_visible(False)

        if cnv_source_of:
            out["cnv_source"] = [cnv_source_of.get(s, "none") for s in sel]

        fig.suptitle(f"{title}  (n={len(sel)}; bars red = aggressive, "
                     f"blue = indolent)", fontsize=_fs(11), y=1.0)
        plt.tight_layout()

        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename or "subset_burden.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return out


    # ── clinical timeline (swimmer) beside the genome ────────────────────────

    LOCATION_COLORS = {
        "Sacral":       "#2FA090",
        "Skull base":   "#7B5EA7",
        "Mobile spine": "#E8A33D",
        "Other":        "#8e8e8e",
    }

    @staticmethod
    def _location_group(loc) -> str:
        """Collapse the free-text primary site onto the three anatomical groups."""
        s = str(loc).strip().lower()
        if not s or s in ("nan", "none"):
            return "Other"
        if "sacr" in s or "coccyx" in s:
            return "Sacral"
        if "skull" in s or "clival" in s or "craniocervical" in s or "ccj" in s:
            return "Skull base"
        return "Mobile spine"

    @staticmethod
    def _row_label(lab: str, mrn=None) -> str:
        """Row label; MRN appended only when one was supplied (see plot doc)."""
        ok = mrn is not None and pd.notna(mrn) and str(mrn).strip() != ""
        return f"{lab}\n{mrn}" if ok else str(lab)

    @staticmethod
    def load_survival_meta(
        csv_path: str,
        meta: "pd.DataFrame | None" = None,
        samples: "list | None" = None,
    ) -> pd.DataFrame:
        """
        Parse the clinical survival sheet into one row per SEQUENCING id.

        The sheet is one row per PATIENT, and a patient who contributed several
        samples carries all of their ids semicolon-joined in `sample` — so that
        column has to be exploded before the sheet will join to anything keyed
        by sample. Rows with no sequencing id are registry-only patients; they
        are dropped, because there is no genome to put beside their timeline.

        Every offset — recurrence, metastasis — is MONTHS FROM DIAGNOSIS. Where
        the diagnosis date is missing the offsets are dropped rather than
        re-measured from first surgery: mixing the two baselines inside one
        swimmer plot would silently put markers on different clocks. `os_months`
        is the sheet's own OS_months, so a row can carry an OS bar with no
        recurrence markers on it — that is missing dates, not absent disease.

        `meta` (the sample metadata) adds a `label` column so the result can
        also be looked up by the p<N> label the rest of the notebook uses.

        Returns a frame indexed by raw sample id with: patient_row, label,
        os_months, os_event, vital_status, n_recurrence, rec_months (list),
        met_month, metastasis, primary_location, location_group.
        """
        if not os.path.exists(csv_path):
            print(f"[WARN] Survival sheet not found: {csv_path}")
            return pd.DataFrame()

        s = pd.read_csv(csv_path)
        id_col = "sample" if "sample" in s.columns else s.columns[0]
        s = s[s[id_col].notna()
              & s[id_col].astype(str).str.contains("ONTWGS")].copy()
        if s.empty:
            print("[WARN] No sequenced patients in the survival sheet.")
            return pd.DataFrame()

        s["_ids"] = s[id_col].astype(str).str.split(r"\s*;\s*")
        s = s.explode("_ids")
        s["_ids"] = s["_ids"].str.strip()
        s = s[s["_ids"].str.startswith("ONTWGS")]

        def _dt(col):
            return (pd.to_datetime(s[col], errors="coerce")
                    if col in s.columns else pd.Series(pd.NaT, index=s.index))

        dx = _dt("Diagnosis_date")

        def _months(col):
            return ((_dt(col) - dx).dt.days / 30.4375).round(2)

        rec_cols = sorted(c for c in s.columns
                          if re.fullmatch(r"Recurrence_\d+_date", c))
        rec = pd.DataFrame({c: _months(c) for c in rec_cols}, index=s.index)
        if "First_recurrence_date" in s.columns:
            rec["_first"] = _months("First_recurrence_date")

        out = pd.DataFrame(index=pd.Index(s["_ids"].to_numpy(), name="sample_id"))
        out["patient_row"] = s[id_col].to_numpy()
        out["os_months"] = pd.to_numeric(
            s.get("OS_months"), errors="coerce").to_numpy()
        out["os_event"] = pd.to_numeric(
            s.get("OS_event(1=died)"), errors="coerce").fillna(0).astype(int).to_numpy()
        out["vital_status"] = s.get(
            "Vital_status", pd.Series(index=s.index)).to_numpy()
        out["n_recurrence"] = pd.to_numeric(
            s.get("Number_of_recurrences"), errors="coerce").to_numpy()
        out["metastasis"] = (s.get("Metastasis", pd.Series(index=s.index))
                             .astype(str).str.strip().str.upper().eq("Y").to_numpy())
        out["met_month"] = _months("Metastasis_date").to_numpy()
        out["primary_location"] = s.get(
            "Primary_location", pd.Series(index=s.index)).to_numpy()
        out["location_group"] = [PatientMethods._location_group(v)
                                 for v in out["primary_location"]]
        out["rec_months"] = [sorted({round(float(v), 2) for v in row
                                     if pd.notna(v) and float(v) >= 0})
                             for row in rec.to_numpy()]

        out = PatientMethods._apply_sheet_fixes(out, "survival sheet")

        if meta is not None and not meta.empty and "label" in meta.columns:
            out["label"] = [meta["label"].get(i) for i in out.index]
        if samples is not None:
            want = set(samples)
            missing = [i for i in samples if i not in set(out.index)]
            if missing:
                print(f"[INFO] no survival row for {len(missing)} sample(s): "
                      f"{', '.join(missing[:6])}"
                      + (" ..." if len(missing) > 6 else ""))
            out = out.loc[[i for i in out.index if i in want]]

        print(f"Survival sheet: {len(out)} sequenced sample(s), "
              f"{out['patient_row'].nunique()} patient(s); "
              f"{int(out['os_event'].sum())} death event(s)")
        return out

    @staticmethod
    def _swimmer_legend() -> tuple:
        """
        Handles for the timeline marks, built from nothing rather than
        harvested off the first row — whether a mark gets a legend entry must
        not depend on whether the top patient happens to have died.
        """
        h = [plt.Line2D([0], [0], color="#b5202a", lw=3),
             plt.Line2D([0], [0], color="#555555", marker=">", lw=0,
                        markersize=7),
             plt.Line2D([0], [0], color="#111111", marker="D", lw=0,
                        markersize=5.5, markeredgecolor="white"),
             plt.Line2D([0], [0], color="#f5d000", marker="*", lw=0,
                        markersize=10, markeredgecolor="#333333")]
        return h, ["Dead (event)", "Alive (ongoing)", "Recurrence", "Metastasis"]

    @staticmethod
    def _surv_index(meta: pd.DataFrame, surv: pd.DataFrame, sel: list) -> dict:
        """label -> that sample's survival row, whichever way `surv` is keyed."""
        by_key = {}
        if surv is not None and not surv.empty:
            for i, r in surv.iterrows():
                by_key[i] = r
                if "label" in surv.columns and pd.notna(r.get("label")):
                    by_key[r["label"]] = r
        raw_of = dict(zip(sel, PatientMethods._raw_ids(meta, sel)))
        return {s: by_key.get(s, by_key.get(raw_of.get(s, s))) for s in sel}

    @staticmethod
    def _draw_swimmer(ax, row):
        """One clinical timeline: OS bar, recurrence diamonds, death cap."""
        col = PatientMethods.LOCATION_COLORS.get(
            row.get("location_group", "Other"), "#8e8e8e")
        os_m = row.get("os_months")
        if not pd.notna(os_m) or os_m <= 0:
            ax.text(0.5, 0.5, "no follow-up", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_fs(6), color="#999999")
            return
        ax.barh(0, os_m, height=0.55, color=col, zorder=2)

        if int(row.get("os_event", 0) or 0) == 1:
            ax.plot([os_m, os_m], [-0.36, 0.36], color="#b5202a", lw=3.0,
                    solid_capstyle="butt", zorder=4)
        else:
            ax.plot([os_m], [0], marker=">", markersize=7, color=col,
                    markeredgecolor=col, zorder=4)

        for m in (row.get("rec_months") or []):
            if m <= os_m:
                ax.plot([m], [0], marker="D", markersize=5.5, color="#111111",
                        markeredgecolor="white", markeredgewidth=0.6, zorder=5)

        mm = row.get("met_month")
        if pd.notna(mm) and 0 <= mm <= os_m:
            ax.plot([mm], [0], marker="*", markersize=10, color="#f5d000",
                    markeredgecolor="#333333", markeredgewidth=0.5, zorder=6)

        n_rec = row.get("n_recurrence")
        ax.text(1.015, 0.5, "–" if pd.isna(n_rec) else f"{int(n_rec)}",
                transform=ax.transAxes, ha="left", va="center", fontsize=_fs(7))

    @staticmethod
    def plot_variant_swimmer(
        samples: list,
        meta: pd.DataFrame,
        surv: pd.DataFrame,
        kind: str = "CNV",
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        genes: "list | None" = None,
        sv_min_len: int = 10_000,
        sv_exclude_types: "tuple" = ("INV", "INVDUP", "DEL/INV"),
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_baseline: str = "ploidy",
        cnv_source_of: "dict | None" = None,
        snv_bin_bp: int = 10_000_000,
        gene_window_bp: int = 100_000,
        mask_unmappable: bool = True,
        cn_max: "float | None" = None,
        max_months: "float | None" = None,
        row_height: float = 1.45,
        mrn_of: "dict | None" = None,
        title: "str | None" = None,
        ytick_scale: float = 1.0,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        One row per patient: the genome-wide profile from ONE caller on the
        left, that patient's clinical course on the right.

        `ytick_scale` multiplies the left panel's y tick labels — the copy
        number and ploidy-baseline numbers — on top of FONT_SCALE. They sit in
        a narrow gutter beside a very wide panel, so they are the smallest text
        in the figure and the first to fail at poster size.

        Deliberately one caller per figure. SNV counts, SV breakpoints and
        copy-number segments are different measurements on different scales;
        stacking them into a single panel invites reading a tall SNV row and a
        busy SV row as the same kind of statement. `kind` picks the panel:

          "SNV"  variants per `snv_bin_bp` bin, SNV and INDEL stacked
          "SV"   breakpoints as stems, height = log10 span, coloured by type,
                 inversions excluded (see build_summary_table for why)
          "CNV"  copy-number step profile coloured against the sample's own
                 ploidy, with that ploidy drawn as a dashed baseline

        Key genes get a dotted rule through every row and a caret in the rows
        where this caller hits them, so one locus can be followed down the
        column and read across into the timeline.

        The timeline is months from diagnosis: bar = overall survival, colour =
        primary site, diamonds = recurrences, star = metastasis, red cap =
        death, arrow = alive at last follow-up, number at the right = the
        recurrence count from the sheet.

        `mrn_of` maps sample id OR patient label -> MRN and puts it under the
        row label. **MRN IS PHI**: use it only for a figure that stays on your
        own machine, and write those to a save_dir outside the shared plot
        folder so they cannot be swept into the HTML report or a poster.

        Returns the per-row table that was drawn.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        kind = kind.upper()
        if kind not in ("SNV", "SV", "CNV"):
            raise ValueError("kind must be one of SNV, SV, CNV")

        sel = [s for s in samples if s in set(meta.index)]
        if not sel:
            print("[WARN] None of the requested samples are in the metadata.")
            return pd.DataFrame()

        # the survival table may be keyed by raw id or by label — accept either
        # meta is passed around indexed either way — _raw_ids handles both
        raw_of = dict(zip(sel, PatientMethods._raw_ids(meta, sel)))
        surv_of = PatientMethods._surv_index(meta, surv, sel)

        def _surv_row(lab):
            return surv_of.get(lab)

        def _mrn(lab):
            return (mrn_of.get(lab, mrn_of.get(raw_of.get(lab, lab)))
                    if mrn_of else None)

        chroms = [c for c in CHROMS if c in HG38_CHROM_SIZES]
        offsets, ticks, total = CohortMethods._genome_offsets(chroms)

        def _nc(c):
            return CohortMethods._norm_chrom(c)

        def _gx(co):
            return offsets.get(_nc(co["chrom"]), 0) + (co["start"] + co["end"]) / 2

        if genes is None:
            genes = ["TBXT", "CDKN2A/B", "PIK3CA", "PTEN", "PBRM1", "SETD2",
                     "TP53", "NF1", "SMARCB1", "EGFR"]
        gene_pos = {g: CHORDOMA_GENE_COORDS[g] for g in genes
                    if g in CHORDOMA_GENE_COORDS}
        if [g for g in genes if g not in CHORDOMA_GENE_COORDS]:
            print("[INFO] no coordinates for "
                  + ", ".join(g for g in genes if g not in CHORDOMA_GENE_COORDS)
                  + " — skipped")

        # ── prepare once, so the axes loop stays readable ────────────────────
        if kind == "SV" and sv_all is not None and not sv_all.empty:
            sv_all = sv_all[(sv_all["svtype"] == "BND")
                            | (sv_all["svlen"].abs() >= sv_min_len)]
            sv_all = sv_all[~sv_all["svtype"].isin(sv_exclude_types)]
        g_thr = l_thr = {}
        if kind == "CNV" and cnv_all is not None and not cnv_all.empty:
            g_thr, l_thr = CohortMethods._cnv_thresholds(
                cnv_all, "cn", cnv_gain, cnv_loss, cnv_baseline, verbose=False)
        # Fixed y scale of 6, not derived from the data. The scale is SHARED
        # across rows, so scaling to the maximum lets one segment flatten every
        # row: on the six flagged patients the maximum is CN 63 and all six
        # rows collapse onto the axis.
        #
        # Nothing real is lost. Every segment above 6 in this cohort is a
        # 150-250 kb pericentromeric window — chr16:46,000,001-46,250,001 in
        # five of the six flagged samples (CN 7 to 63), chr10:42,000,001 in two
        # more. Satellite repeat, mismapped, not amplification: 9 of 810
        # segments and 0.022% of the genome, sub-pixel at this scale. The rows
        # that clip are annotated with how many segments went over.
        cn_max = cn_max or 6.0

        if max_months is None:
            vals = [r.get("os_months") for r in (_surv_row(s) for s in sel)
                    if r is not None]
            vals = [v for v in vals if pd.notna(v)]
            max_months = float(max(vals)) * 1.14 if vals else 120.0

        n = len(sel)
        fig, axes = plt.subplots(
            n, 2, sharex="col", squeeze=False,
            figsize=(17.0, row_height * n + 2.2),
            gridspec_kw={"width_ratios": [5.0, 1.45], "wspace": 0.14,
                         "hspace": 0.25})

        rows_out, sv_types_seen = [], set()
        for i, lab in enumerate(sel):
            axL, axR = axes[i, 0], axes[i, 1]
            CohortMethods._draw_genome_axis(axL, chroms, offsets, total)
            hit, count, note = set(), 0, ""

            if kind == "SNV":
                d = (snv_all[snv_all["sample"] == lab]
                     if snv_all is not None and not snv_all.empty
                     else pd.DataFrame())
                count = len(d)
                if d.empty:
                    note = "no SNV data"
                else:
                    cn_ = d["chrom"].map(_nc)
                    off = cn_.map(offsets).to_numpy(float)
                    ok = np.isfinite(off)
                    x = off[ok] + d["pos"].to_numpy(float)[ok]
                    bins = np.arange(0, total + snv_bin_bp, snv_bin_bp)
                    if "vtype" in d.columns:
                        vt = d["vtype"].to_numpy()[ok]
                        axL.hist([x[vt == "SNV"], x[vt == "INDEL"]], bins=bins,
                                 stacked=True, color=["#4da6ff", "#003f8a"],
                                 histtype="stepfilled", linewidth=0, zorder=2,
                                 label=["SNV", "INDEL"] if i == 0 else None)
                    else:
                        axL.hist(x, bins=bins, color="#4da6ff",
                                 histtype="stepfilled", linewidth=0, zorder=2)
                    for g, co in gene_pos.items():
                        if bool(((cn_ == _nc(co["chrom"]))
                                 & (d["pos"] >= co["start"] - gene_window_bp)
                                 & (d["pos"] <= co["end"] + gene_window_bp)).any()):
                            hit.add(g)
                    axL.tick_params(axis="y", labelsize=_fs(6) * ytick_scale)

            elif kind == "SV":
                d = (sv_all[sv_all["sample"] == lab]
                     if sv_all is not None and not sv_all.empty
                     else pd.DataFrame())
                count = len(d)
                if d.empty:
                    note = "no SV data"
                for r in d.itertuples():
                    span = abs(r.svlen) if pd.notna(r.svlen) else 0
                    h = (1.0 if r.svtype == "BND"
                         else float(np.clip(np.log10(max(span, 10)) / 8.0, 0.12, 1.0)))
                    sv_types_seen.add(r.svtype)
                    for ch, pos in ((r.chrom, r.start), (r.chrom2, r.end)):
                        c = _nc(ch)
                        if c not in offsets or not pd.notna(pos):
                            continue
                        axL.plot([offsets[c] + pos] * 2, [0, h],
                                 color=SVTYPE_COLORS.get(r.svtype, "#888888"),
                                 lw=0.8, alpha=0.75, zorder=2)
                        for g, co in gene_pos.items():
                            if (c == _nc(co["chrom"])
                                    and co["start"] - gene_window_bp <= pos
                                    <= co["end"] + gene_window_bp):
                                hit.add(g)
                axL.set_ylim(0, 1.12)
                axL.set_yticks([])

            else:                                              # CNV
                d = (cnv_all[cnv_all["sample"] == lab]
                     if cnv_all is not None and not cnv_all.empty
                     else pd.DataFrame())
                count = len(d)
                if d.empty:
                    note = "no CNV data"
                base = (float(d["ploidy"].iloc[0])
                        if ("ploidy" in d.columns and len(d)
                            and pd.notna(d["ploidy"].iloc[0])) else 2.0)
                gt, lt = g_thr.get(lab, cnv_gain), l_thr.get(lab, cnv_loss)
                axL.axhline(base, color="#999999", lw=0.7, ls="--", zorder=1)
                for r in d.itertuples():
                    c = _nc(r.chrom)
                    if c not in offsets or not pd.notna(r.cn):
                        continue
                    col = ("#c1272d" if r.cn >= gt
                           else "#2c7fb8" if r.cn <= lt else "#777777")
                    # subtract the unmappable blocks instead of dropping the
                    # segment: these are often arm-length, so an overlap test
                    # would delete whole chromosomes from the row
                    pieces = (mask_subtract(r.chrom, r.start, r.end)
                              if mask_unmappable else [(r.start, r.end)])
                    y = min(r.cn, cn_max)
                    for ps, pe in pieces:
                        axL.plot([offsets[c] + ps, offsets[c] + pe], [y, y],
                                 color=col, lw=2.0, solid_capstyle="butt", zorder=3)
                    for g, co in gene_pos.items():
                        if (c == _nc(co["chrom"]) and r.end > co["start"]
                                and r.start < co["end"]
                                and (r.cn >= gt or r.cn <= lt)):
                            hit.add(g)
                axL.set_ylim(0, cn_max * 1.10)
                axL.set_yticks(sorted({0, round(base, 1), cn_max}))
                axL.tick_params(axis="y", labelsize=_fs(6) * ytick_scale)

            if note:
                axL.text(0.5, 0.5, note, transform=axL.transAxes, ha="center",
                         va="center", fontsize=_fs(8), color="#999999")

            src = (cnv_source_of or {}).get(lab) if kind == "CNV" else None
            axL.set_ylabel(PatientMethods._row_label(lab, _mrn(lab))
                           + (f"\n({src})" if src else ""),
                           fontsize=_fs(8.5), rotation=0, ha="right",
                           va="center", labelpad=10)
            axL.spines[["top", "right"]].set_visible(False)

            # key-gene rules, with a caret where this caller hits the gene
            top = axL.get_ylim()[1]
            for g, co in gene_pos.items():
                x = _gx(co)
                axL.axvline(x, color="#b8860b", lw=0.6, ls=":", alpha=0.7, zorder=1)
                if g in hit:
                    axL.plot([x], [top * 0.93], marker="v", markersize=5,
                             color="#b8860b", zorder=6, clip_on=False)

            # ── clinical timeline ────────────────────────────────────────────
            r = _surv_row(lab)
            if r is None:
                axR.text(0.5, 0.5, "no clinical row", transform=axR.transAxes,
                         ha="center", va="center", fontsize=_fs(6), color="#999999")
            else:
                PatientMethods._draw_swimmer(axR, r)
            axR.set_xlim(0, max_months)
            axR.set_ylim(-0.55, 0.55)
            axR.set_yticks([])
            axR.spines[["top", "right", "left"]].set_visible(False)
            axR.grid(axis="x", color="#e6e6e6", lw=0.5)
            axR.set_axisbelow(True)

            rows_out.append({
                "label": lab,
                "sample_id": raw_of.get(lab, lab),
                **({"MRN": _mrn(lab)} if mrn_of else {}),
                f"n_{kind.lower()}": count,
                "genes_hit": ";".join(sorted(hit)),
                "os_months": None if r is None else r.get("os_months"),
                "os_event": None if r is None else r.get("os_event"),
                "n_recurrence": None if r is None else r.get("n_recurrence"),
                "location_group": None if r is None else r.get("location_group"),
            })

        # Gene names along the top, once. Staggered over two lines: PBRM1
        # and SETD2 are 5 Mb apart on chr3 and TP53/NF1 24 Mb apart on chr17,
        # which is under a label width at this figure size.
        for j, (g, co) in enumerate(sorted(gene_pos.items(),
                                           key=lambda kv: _gx(kv[1]))):
            axes[0, 0].annotate(g, xy=(_gx(co), 1.0),
                                xycoords=("data", "axes fraction"),
                                xytext=(0, 6 + 26 * (j % 2)),
                                textcoords="offset points",
                                rotation=90, ha="center", va="bottom",
                                fontsize=_fs(6.5), color="#8a6508")

        axes[-1, 0].set_xticks(ticks)
        axes[-1, 0].set_xticklabels(chroms, fontsize=_fs(7))
        axes[-1, 0].set_xlabel("genome position", fontsize=_fs(9))
        axes[-1, 1].set_xlabel("months from diagnosis", fontsize=_fs(9))
        axes[-1, 1].tick_params(axis="x", labelsize=_fs(7))
        axes[0, 1].set_title("Rec.", fontsize=_fs(7), loc="right", pad=3)

        ylab = {"SNV": f"variants per {snv_bin_bp // 1_000_000} Mb",
                "SV": "breakpoints, stem height = log10 span (BND full)",
                "CNV": "copy number"}[kind]
        sub = {"SNV": "ClairS-TO PASS, VAF 0.1–0.6",
               "SV": f"sniffles ≥{sv_min_len // 1000} kb, inversions excluded",
               "CNV": f"vs each sample's own ploidy ({cnv_baseline} baseline)"}[kind]
        # Reserve the header in INCHES, not as a fraction of the figure.
        # Above the top row sit the rotated gene labels (~0.85 in for a name
        # like CDKN2A/B) and then the two-line title (~0.45 in) — both fixed
        # sizes regardless of how many rows there are. matplotlib's default
        # top=0.88 leaves 1.3 in on a six-row figure, which is why it looked
        # right there, and only 0.6 in on a two-row one, where the title's
        # second line lands on the gene labels. 1.62 in clears the longest
        # panel names (CDKN2A/B, SMARCB1) at every row count.
        _fig_h = row_height * n + 2.2
        _header = 1.62
        if 1.0 - _header / _fig_h < 0.70:      # one- or two-row figures
            _fig_h = _header / 0.30            # keep a usable plotting area
            fig.set_size_inches(17.0, _fig_h)
        fig.subplots_adjust(top=1.0 - _header / _fig_h)
        fig.suptitle((title or f"{kind} landscape and clinical course "
                               f"— {len(sel)} patient(s)")
                     + f"\nleft: {ylab} · {sub}    right: overall survival, "
                       f"months from diagnosis",
                     fontsize=_fs(12), y=0.995, va="top")

        handles, labels_ = PatientMethods._swimmer_legend()
        if kind == "SV":
            extra = sorted(sv_types_seen)
            handles = [plt.Line2D([0], [0], color=SVTYPE_COLORS.get(t, "#888888"),
                                  lw=2) for t in extra] + handles
            labels_ = extra + labels_
        elif kind == "SNV":
            h2, l2 = axes[0, 0].get_legend_handles_labels()
            handles, labels_ = h2 + handles, l2 + labels_
        else:
            handles = [plt.Line2D([0], [0], color=c, lw=2)
                       for c in ("#c1272d", "#777777", "#2c7fb8")] + handles
            labels_ = ["gain", "neutral", "loss"] + labels_
        sites = [k for k in PatientMethods.LOCATION_COLORS if k != "Other"]
        handles += [mpatches.Patch(facecolor=PatientMethods.LOCATION_COLORS[k])
                    for k in sites]
        labels_ += sites
        fig.legend(handles, labels_, loc="upper center", frameon=False,
                   ncol=min(9, len(labels_)), fontsize=_fs(8),
                   bbox_to_anchor=(0.5, 0.035))

        out = pd.DataFrame(rows_out)
        if actual_dir is not None:
            fpath = os.path.join(actual_dir, filename
                                 or f"swimmer_{kind.lower()}"
                                    f"{'_mrn' if mrn_of else ''}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return out


    # ── methylation browser beside the clinical course ───────────────────────

    @staticmethod
    def _meth_region(bed_path: str, chrom: str, start: int, end: int,
                     browser=None, min_coverage: int = 1) -> pd.DataFrame:
        """
        CpG sites in one window: pos, freq (0-100), coverage.

        Prefers the tabix index, falling back to the browser's plain scan. The
        pileups are ~460 MB gzipped and the browser reads them start to finish
        for every window; with a .tbi sitting next to the file that is a minute
        per sample per locus thrown away, and this figure asks for one window
        per sample.
        """
        if not bed_path or not os.path.exists(bed_path):
            print(f"[WARN] no such pileup: {bed_path}")
            return pd.DataFrame(columns=["pos", "freq", "coverage"])
        c = str(chrom)
        for want in (f"chr{c}" if not c.startswith("chr") else c,
                     c[3:] if c.startswith("chr") else c):
            try:
                import pysam
                with pysam.TabixFile(bed_path) as tb:
                    if want not in tb.contigs:
                        continue
                    rows = []
                    for line in tb.fetch(want, max(0, start), end):
                        f = line.split("\t")
                        if len(f) < 11:
                            continue
                        try:
                            cov, freq = int(f[9]), float(f[10])
                        except ValueError:
                            continue
                        if cov >= min_coverage:
                            rows.append((int(f[1]), freq, cov))
                    return pd.DataFrame(rows, columns=["pos", "freq", "coverage"])
            except (ImportError, OSError, ValueError):
                break
        if browser is not None:
            return browser.load_region(bed_path, c, start, end)
        print(f"[WARN] cannot read {os.path.basename(bed_path)} — no tabix index "
              f"and no browser module to fall back on")
        return pd.DataFrame(columns=["pos", "freq", "coverage"])

    @staticmethod
    def plot_methylation_swimmer(
        samples: list,
        meta: pd.DataFrame,
        surv: pd.DataFrame,
        gene: "str | dict" = "TBXT",
        flank_bp: int = 2_000_000,
        browser=None,
        bed_of: "dict | None" = None,
        smooth_sites: "int | None" = None,
        min_coverage: int = 1,
        point_size: float = 2.0,
        skip_missing: bool = True,
        row_height: float = 1.45,
        max_months: "float | None" = None,
        mrn_of: "dict | None" = None,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        The methylation browser as swimmer rows: one sample per row, CpG
        methylation across a locus on the left, that patient's clinical course
        on the right.

        The methylation panel is a LOCUS view, not a genome-wide one like the
        SNV/SV/CNV figures — a bedMethyl pileup is read per window, so "the
        whole genome" would mean reading 460 MB per sample per row. Pass `gene`
        (any key in CHORDOMA_GENE_COORDS, TBXT included) or an explicit
        {"chrom","start","end"} dict.

        `browser` is the methylation_browser_chordoma module, used for exactly
        three things: finding a sample's pileup, its CpG island table, and its
        colours. Call `browser.configure(run_bases=...)` first to point it at
        the local modkit trees. The dependency runs one way only — the browser
        must not import this library, or it stops working on the cluster.

        `bed_of` overrides discovery with an explicit {sample or label: path},
        which is how to reach a pileup whose directory is misnamed.

        `skip_missing` drops samples with no pileup and says which; False keeps
        them as empty rows instead.

        `mrn_of` maps sample id OR patient label -> MRN and puts it under the
        row label, exactly as plot_variant_swimmer does. **MRN IS PHI**: write
        those figures to a save_dir outside the shared plot folder.

        Returns a row per sample: n_cpg, mean/median methylation in the window,
        mean coverage, and the clinical fields.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        sel = [s for s in samples if s in set(meta.index)]
        if not sel:
            print("[WARN] None of the requested samples are in the metadata.")
            return pd.DataFrame()

        gname, co = CohortMethods.resolve_gene(gene)
        chrom = CohortMethods._norm_chrom(co["chrom"])
        win_s, win_e = max(1, co["start"] - flank_bp), co["end"] + flank_bp

        raw_of = dict(zip(sel, PatientMethods._raw_ids(meta, sel)))
        surv_of = PatientMethods._surv_index(meta, surv, sel)

        def _bed(lab):
            # An override that no longer resolves means the file moved — fall
            # back to discovery and say so, rather than handing a dead path to
            # the reader and dying on open(). These entries exist to work
            # around misnamed directories, so they go stale exactly when the
            # directory gets fixed.
            if bed_of:
                p = bed_of.get(lab, bed_of.get(raw_of.get(lab, lab)))
                if p and os.path.exists(p):
                    return p
                if p:
                    print(f"[WARN] bed_of path for {lab} no longer exists "
                          f"({p}) — falling back to discovery")
            return (browser.find_bedmethyl(raw_of.get(lab, lab))
                    if browser is not None else None)

        beds = {s: _bed(s) for s in sel}
        missing = [s for s in sel if not beds[s]]
        if missing:
            print(f"[INFO] no pileup for {len(missing)} sample(s): "
                  f"{', '.join(missing)}"
                  + ("  — dropped" if skip_missing else "  — drawn empty"))
            if skip_missing:
                sel = [s for s in sel if beds[s]]
        if not sel:
            print("[WARN] No sample in this selection has methylation data.")
            return pd.DataFrame()

        if smooth_sites is None:
            smooth_sites = (browser.smooth_sites(flank_bp)
                            if browser is not None
                            else (20 if flank_bp < 100_000 else 1000))
        col_m = getattr(browser, "COL_M", "#5b9bd5")
        col_cpg = getattr(browser, "COL_CPG", "#2ecc71")
        col_gene = getattr(browser, "COL_GENE", "#f2c14e")

        islands = []
        if browser is not None:
            try:
                islands = browser.fetch_cpg_islands(chrom, win_s, win_e)
            except Exception as e:                       # noqa: BLE001
                print(f"[WARN] CpG islands unavailable: {e}")

        if max_months is None:
            v = [r.get("os_months") for r in surv_of.values() if r is not None]
            v = [x for x in v if pd.notna(x)]
            max_months = float(max(v)) * 1.14 if v else 120.0

        n = len(sel)
        fig, axes = plt.subplots(
            n, 2, sharex="col", squeeze=False,
            figsize=(17.0, row_height * n + 2.2),
            gridspec_kw={"width_ratios": [5.0, 1.45], "wspace": 0.14,
                         "hspace": 0.25})

        rows_out = []
        for i, lab in enumerate(sel):
            axL, axR = axes[i, 0], axes[i, 1]
            d = (PatientMethods._meth_region(beds[lab], chrom, win_s, win_e,
                                             browser=browser,
                                             min_coverage=min_coverage)
                 if beds[lab] else pd.DataFrame())

            for s0, e0 in islands:
                axL.axvspan(s0, e0, color=col_cpg, alpha=0.22, lw=0, zorder=0)
            if browser is not None:
                browser.mark_gene(axL, co["start"], co["end"],
                                  win_s, win_e, flank_bp)
            else:
                for x in (co["start"], co["end"]):
                    axL.axvline(x, color=col_gene, lw=1.0, ls="--", zorder=0)

            if d.empty:
                axL.text(0.5, 0.5, "no methylation data", transform=axL.transAxes,
                         ha="center", va="center", fontsize=_fs(8), color="#999999")
                stats = dict(n_cpg=0, mean_meth=np.nan, median_meth=np.nan,
                             mean_cov=np.nan)
            else:
                d = d.sort_values("pos")
                axL.scatter(d["pos"], d["freq"], s=point_size, alpha=0.35,
                            color=col_m, linewidths=0, zorder=2)
                if len(d) >= smooth_sites:
                    axL.plot(d["pos"],
                             d["freq"].rolling(smooth_sites, center=True,
                                               min_periods=max(1, smooth_sites // 4)
                                               ).mean(),
                             color="#1f4e79", lw=1.2, zorder=3)
                stats = dict(n_cpg=len(d), mean_meth=float(d["freq"].mean()),
                             median_meth=float(d["freq"].median()),
                             mean_cov=float(d["coverage"].mean()))
                # boxed: at 2 Mb the scatter saturates the top of the panel
                # and unboxed text landed inside it
                axL.text(0.998, 0.94,
                         f"{stats['n_cpg']:,} CpG · mean {stats['mean_meth']:.0f}% "
                         f"· cov {stats['mean_cov']:.0f}x",
                         transform=axL.transAxes, ha="right", va="top",
                         fontsize=_fs(6), color="#333333",
                         bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.80, pad=1.5))

            axL.set_xlim(win_s, win_e)
            axL.set_ylim(-4, 104)
            axL.set_yticks([0, 50, 100])
            axL.tick_params(axis="y", labelsize=_fs(6))
            axL.set_ylabel(
                PatientMethods._row_label(
                    lab, (mrn_of.get(lab, mrn_of.get(raw_of.get(lab, lab)))
                          if mrn_of else None)),
                fontsize=_fs(8.5), rotation=0, ha="right", va="center",
                labelpad=10)
            axL.spines[["top", "right"]].set_visible(False)

            r = surv_of.get(lab)
            if r is None:
                axR.text(0.5, 0.5, "no clinical row", transform=axR.transAxes,
                         ha="center", va="center", fontsize=_fs(6), color="#999999")
            else:
                PatientMethods._draw_swimmer(axR, r)
            axR.set_xlim(0, max_months)
            axR.set_ylim(-0.55, 0.55)
            axR.set_yticks([])
            axR.spines[["top", "right", "left"]].set_visible(False)
            axR.grid(axis="x", color="#e6e6e6", lw=0.5)
            axR.set_axisbelow(True)

            rows_out.append({
                "label": lab, "sample_id": raw_of.get(lab, lab),
                **({"MRN": mrn_of.get(lab, mrn_of.get(raw_of.get(lab, lab)))}
                   if mrn_of else {}),
                **stats,
                "os_months": None if r is None else r.get("os_months"),
                "os_event": None if r is None else r.get("os_event"),
                "n_recurrence": None if r is None else r.get("n_recurrence"),
                "location_group": None if r is None else r.get("location_group"),
            })

        axes[-1, 0].set_xlabel(f"chr{chrom} position (bp)", fontsize=_fs(9))
        axes[-1, 0].tick_params(axis="x", labelsize=_fs(7))
        axes[-1, 1].set_xlabel("months from diagnosis", fontsize=_fs(9))
        axes[-1, 1].tick_params(axis="x", labelsize=_fs(7))
        axes[0, 1].set_title("Rec.", fontsize=_fs(7), loc="right", pad=3)

        span = f"{gname} ±{flank_bp / 1e6:g} Mb" if flank_bp else gname
        fig.suptitle((title or f"Methylation at {span} and clinical course "
                               f"— {n} sample(s)")
                     + f"\nleft: CpG methylation %, rolling mean over "
                       f"{smooth_sites} sites, CpG islands green    "
                       f"right: overall survival, months from diagnosis",
                     fontsize=_fs(12), y=1.045)

        handles, labels_ = PatientMethods._swimmer_legend()
        handles = [plt.Line2D([0], [0], color=col_m, marker="o", lw=0,
                              markersize=4),
                   plt.Line2D([0], [0], color="#1f4e79", lw=1.5),
                   mpatches.Patch(facecolor=col_cpg, alpha=0.4),
                   mpatches.Patch(facecolor=col_gene, alpha=0.5)] + handles
        labels_ = ["CpG site", "rolling mean", "CpG island", gname] + labels_
        sites = [k for k in PatientMethods.LOCATION_COLORS if k != "Other"]
        handles += [mpatches.Patch(facecolor=PatientMethods.LOCATION_COLORS[k])
                    for k in sites]
        labels_ += sites
        fig.legend(handles, labels_, loc="upper center", frameon=False,
                   ncol=min(8, len(labels_)), fontsize=_fs(8),
                   bbox_to_anchor=(0.5, 0.035))

        out = pd.DataFrame(rows_out)
        if actual_dir is not None:
            tag = str(gname).replace("/", "_")
            fpath = os.path.join(
                actual_dir, filename or f"swimmer_meth_{tag.lower()}"
                                        f"{'_mrn' if mrn_of else ''}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return out

    @staticmethod
    def plot_tss_methylation_profile(
        samples: list,
        meta: pd.DataFrame,
        gene: "str | dict" = "TERT",
        tss: "int | None" = None,
        strand: str = "-",
        upstream_bp: int = 6_000,
        downstream_bp: int = 6_000,
        regions: "dict | None" = None,
        browser=None,
        bed_of: "dict | None" = None,
        min_coverage: int = 5,
        smooth_sites: int = 15,
        skip_missing: bool = True,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        CpG methylation around a transcription start site, oriented by strand,
        with named sub-regions called out.

        **Strand matters here and nowhere else in this library.** Every other
        gene lookup uses the body interval (start, end), which is identical on
        either strand. A TSS is not: for a minus-strand gene the TSS is the
        HIGHER coordinate and "upstream" runs toward larger positions. Getting
        this backwards silently reports 3'-end methylation as promoter
        methylation — for TBXT that is the difference between 4% and 70%.

        `tss` defaults to the strand-correct end of `gene`'s span. `strand`
        must be given because CHORDOMA_GENE_COORDS carries no strand field;
        minus-strand entries in that table include TBXT, FN1, CDKN2A/B, PTEN,
        NF1, SMARCB1 and TERT.

        The x axis is distance from the TSS in transcriptional orientation:
        negative is upstream (promoter, enhancers), positive is into the gene
        body. `regions` is {label: (genomic_start, genomic_end)} and is drawn
        as shaded bands on the left panel and as one row per region on the
        right, so a region whose methylation varies across samples is visible
        as a wide scatter of dots rather than a number in a table.

        Defaults describe TERT: THOR (the TERT Hypermethylated Oncological
        Region, ~TSS+21..+453 upstream), where methylation tracks TERT
        REACTIVATION rather than silencing, plus the proximal promoter and the
        5' gene body for contrast.

        Returns one row per sample per region: mean/median methylation, CpG
        count and mean coverage.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        sel = [s for s in samples if s in set(meta.index)]
        if not sel:
            print("[WARN] None of the requested samples are in the metadata.")
            return pd.DataFrame()

        gname, co = CohortMethods.resolve_gene(gene)
        chrom = CohortMethods._norm_chrom(co["chrom"])
        if strand not in ("+", "-"):
            raise ValueError(f"strand must be '+' or '-', got {strand!r}")
        if tss is None:
            tss = co["start"] if strand == "+" else co["end"]

        # Window in genomic coordinates; upstream is to the LEFT on a plus
        # strand gene and to the RIGHT on a minus strand one.
        if strand == "+":
            win_s, win_e = tss - upstream_bp, tss + downstream_bp
        else:
            win_s, win_e = tss - downstream_bp, tss + upstream_bp
        win_s = max(1, win_s)

        if regions is None and str(gname).upper() == "TERT":
            regions = {
                "THOR":             (1_295_321, 1_295_753),
                "proximal promoter": (tss - 500, tss + 500),
                "5' gene body":     (tss - 5_000, tss - 1_000),
            }
        regions = regions or {}

        raw_of = dict(zip(sel, PatientMethods._raw_ids(meta, sel)))

        def _bed(lab):
            if bed_of:
                p = bed_of.get(lab, bed_of.get(raw_of.get(lab, lab)))
                if p and os.path.exists(p):
                    return p
                if p:
                    print(f"[WARN] bed_of path for {lab} no longer exists "
                          f"({p}) — falling back to discovery")
            if browser is None:
                return None
            return browser.find_bedmethyl(raw_of.get(lab, lab))

        # ── read one window per sample ──────────────────────────────────────
        prof, missing = {}, []
        for lab in sel:
            bed = _bed(lab)
            d = (PatientMethods._meth_region(bed, chrom, win_s, win_e,
                                             browser=browser,
                                             min_coverage=min_coverage)
                 if bed else pd.DataFrame(columns=["pos", "freq", "coverage"]))
            if d.empty:
                missing.append(lab)
                if skip_missing:
                    continue
            prof[lab] = d
        if missing:
            print(f"[INFO] no pileup for {len(missing)} sample(s): "
                  f"{', '.join(map(str, missing))}")
        if not prof:
            print("[WARN] no methylation data in this window.")
            return pd.DataFrame()

        def _dist(pos):
            # transcriptional orientation: negative upstream, positive into
            # the gene body, on either strand
            return (pos - tss) if strand == "+" else (tss - pos)

        rows = []
        for lab, d in prof.items():
            for rname, (rs, re_) in regions.items():
                m = d[(d["pos"] >= min(rs, re_)) & (d["pos"] <= max(rs, re_))]
                rows.append({
                    "sample": lab, "gene": gname, "region": rname,
                    "n_cpg": len(m),
                    "mean_meth": m["freq"].mean() if len(m) else np.nan,
                    "median_meth": m["freq"].median() if len(m) else np.nan,
                    "mean_cov": m["coverage"].mean() if len(m) else np.nan,
                })
        out = pd.DataFrame(rows)

        # ── figure ──────────────────────────────────────────────────────────
        labs = list(prof)
        ncol = 2 if regions else 1
        fig, axes = plt.subplots(
            1, ncol, figsize=(13.5 if regions else 9, 4.6),
            gridspec_kw={"width_ratios": [2.4, 1.0]} if regions else None)
        ax = axes[0] if regions else axes
        cmap = plt.get_cmap("tab10" if len(labs) <= 10 else "tab20")
        colour = {l: cmap(i % cmap.N) for i, l in enumerate(labs)}

        # Region bands. The first region is the one the figure is about, so it
        # gets the accent colour; a 432 bp region inside a 12 kb window is a
        # sliver, and grey-on-grey makes it invisible. Labels are staggered
        # because adjacent regions (THOR abuts the proximal promoter) collide
        # at a single height.
        band = ["#e8a33d", "#9ecae1", "#c7c7c7", "#c9e4c5", "#dcc6e0"]
        for i, (rname, (rs, re_)) in enumerate(regions.items()):
            a, b = sorted((_dist(rs), _dist(re_)))
            c = band[i % len(band)]
            ax.axvspan(a, b, color=c, alpha=0.30 if i else 0.55, lw=0, zorder=0)
            if i == 0:                      # outline the focal region
                for xv in (a, b):
                    ax.axvline(xv, color=c, lw=0.9, zorder=1)
            ax.annotate(rname, xy=((a + b) / 2, 100),
                        xytext=((a + b) / 2, 112 + (i % 2) * 8),
                        ha="center", va="bottom", fontsize=7.5, color="0.2",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.55"))
        ax.axvline(0, color="0.35", lw=1.0, ls="--", zorder=1)
        ax.text(0, -9, "TSS", ha="center", va="top", fontsize=7.5, color="0.35")

        for lab in labs:
            d = prof[lab]
            if d.empty:
                continue
            x = _dist(d["pos"].values).astype(float)
            y = d["freq"].values.astype(float)
            o = np.argsort(x)
            x, y = x[o], y[o]
            ax.scatter(x, y, s=1.6, color=colour[lab], alpha=0.18,
                       lw=0, zorder=2)
            if smooth_sites and len(y) >= smooth_sites:
                ys = pd.Series(y).rolling(smooth_sites, center=True,
                                          min_periods=1).mean().values
                ax.plot(x, ys, color=colour[lab], lw=1.5, label=str(lab),
                        zorder=3)
            else:
                ax.plot(x, y, color=colour[lab], lw=1.0, label=str(lab),
                        zorder=3)
        # Always upstream-left / gene-body-right, whichever strand the gene is
        # on — that is the convention every promoter figure is read against,
        # and deriving the limits from the genomic window flips it for minus
        # strand genes.
        ax.set_xlim(-upstream_bp, downstream_bp)
        ax.set_ylim(-12, 128)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel(f"distance from {gname} TSS (bp) — negative = upstream "
                      f"(promoter), positive = gene body; {strand} strand")
        ax.set_ylabel("CpG methylation (%)")
        ax.legend(fontsize=7, ncol=3, loc="lower left", framealpha=0.9,
                  handlelength=1.3, columnspacing=1.0, borderpad=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        if regions:
            ax2 = axes[1]
            names = list(regions)
            for i, rname in enumerate(names):
                sub = out[out["region"] == rname].dropna(subset=["mean_meth"])
                y = np.full(len(sub), i, dtype=float) + \
                    np.linspace(-0.16, 0.16, len(sub))
                ax2.scatter(sub["mean_meth"], y, s=34,
                            color=[colour[s] for s in sub["sample"]],
                            edgecolor="white", lw=0.6, zorder=3)
                if len(sub):
                    ax2.plot([sub["mean_meth"].min(), sub["mean_meth"].max()],
                             [i, i], color=band[i % len(band)], lw=2.6,
                             alpha=0.55, solid_capstyle="round", zorder=1)
                    sp = sub["mean_meth"].max() - sub["mean_meth"].min()
                    ax2.text(103, i, f"spread {sp:.0f} pts", fontsize=7,
                             va="center", color="0.35")
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=8.5)
            ax2.set_ylim(-0.6, len(names) - 0.4)
            ax2.set_xlim(-3, 118)
            ax2.set_xticks([0, 25, 50, 75, 100])
            ax2.set_xlabel("mean methylation (%)")
            ax2.set_title("per sample, by region", fontsize=9)
            ax2.invert_yaxis()
            ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(title or
                     f"{gname} promoter methylation — {len(labs)} sample(s), "
                     f"CpG resolution, oriented 5'→3'",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        if actual_dir:
            os.makedirs(actual_dir, exist_ok=True)
            tag = str(gname).replace("/", "_").lower()
            fpath = os.path.join(actual_dir,
                                 filename or f"tss_methylation_{tag}.pdf")
            fig.savefig(fpath, bbox_inches="tight")
            print(f"Saved: {fpath}")
            plt.close(fig)
        else:
            plt.show()
        return out

    @staticmethod
    def _perm_spearman_p(x, y, max_exact_n: int = 8, n_perm: int = 200_000,
                         seed: int = 0):
        """
        Two-sided Spearman p by permutation.

        scipy's spearmanr p-value is a t-approximation that assumes enough
        points for asymptotic normality. At n=6 it is wrong by a factor of
        four — it reported p=0.008 for a THOR/aggressiveness correlation whose
        exact p is 0.033 — and a reviewer who checks will find that, so this
        enumerates instead. Exact while n! is affordable, sampled above that;
        the cohort is expected to grow past the exact range.

        Returns (rho, p, kind) where kind is "exact" or "sampled".
        """
        from scipy import stats as _st
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rho = _st.spearmanr(x, y).statistic
        obs = abs(rho)
        n = len(x)
        if n <= max_exact_n:
            import itertools
            hits = tot = 0
            for p in itertools.permutations(y):
                tot += 1
                if abs(_st.spearmanr(x, list(p)).statistic) >= obs - 1e-12:
                    hits += 1
            return float(rho), hits / tot, "exact"
        rng = np.random.default_rng(seed)
        yy = y.copy()
        hits = 0
        for _ in range(n_perm):
            rng.shuffle(yy)
            if abs(_st.spearmanr(x, yy).statistic) >= obs - 1e-12:
                hits += 1
        return float(rho), (hits + 1) / (n_perm + 1), "sampled"

    @staticmethod
    def plot_methylation_aggressiveness(
        profile: pd.DataFrame,
        meta: pd.DataFrame,
        focus_region: str = "THOR",
        weights: "dict | None" = None,
        tier_labels: "dict | None" = None,
        title: "str | None" = None,
        save_dir: "str | None" = None,
        filename: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Methylation of one region against the aggressiveness score, with a
        specificity panel for the other regions measured alongside it.

        `profile` is what plot_tss_methylation_profile returns: one row per
        sample per region. Samples are collapsed to patients by mean, and a
        patient contributing several samples gets its observed RANGE drawn as
        a grey bar labelled (n=k). That bar is the point of the figure as much
        as the trend is: if one tumour's own samples span most of the spread,
        the correlation is not measuring a patient-level property, and hiding
        it invites exactly the question it answers.

        The right panel runs the same test on every other region in `profile`.
        A signal at one region while its neighbours sit near zero is the
        difference between a specific association and a property of the locus.

        p values are permutation tests (see _perm_spearman_p), NOT scipy's
        asymptotic approximation, which is unusable at these sample sizes.
        They are uncorrected and exploratory — say so in the caption, since
        this figure deliberately carries no footnote of its own.

        Returns one row per region: rho, p, the kind of test, and n.
        """
        actual_dir = SNVmethods._gene_save_dir(save_dir)
        if profile is None or profile.empty:
            print("[WARN] empty profile table — nothing to plot.")
            return pd.DataFrame()

        w = dict(PatientMethods.DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)

        # label -> patient, whichever way meta is oriented
        if "patient" not in meta.columns:
            print("[WARN] meta needs a 'patient' column.")
            return pd.DataFrame()
        pat_of = meta["patient"].to_dict()
        prof = profile.copy()
        prof["patient"] = [pat_of.get(s) for s in prof["sample"]]
        if prof["patient"].isna().any():
            miss = sorted(set(prof.loc[prof["patient"].isna(), "sample"]))
            print(f"[WARN] no patient for {len(miss)} sample(s): "
                  f"{', '.join(map(str, miss[:5]))}")
            prof = prof[prof["patient"].notna()]

        flags = meta.groupby("patient").agg(
            dead=("dead", "max"), mets=("metastasis", "max"),
            rec=("n_recurrence", "max"))
        score = (w["dead"] * flags["dead"].astype(float)
                 + w["metastasis"] * flags["mets"].astype(float)
                 + w["recurrence"] * pd.to_numeric(flags["rec"],
                                                   errors="coerce").fillna(0.0))

        regions = [r for r in prof["region"].unique()]
        if focus_region not in regions:
            print(f"[WARN] {focus_region!r} not among {regions}")
            return pd.DataFrame()
        regions = [focus_region] + [r for r in regions if r != focus_region]

        res = []
        for rg in regions:
            v = prof[prof["region"] == rg].groupby("patient")["mean_meth"].mean()
            s = score.reindex(v.index)
            ok = v.notna() & s.notna()
            if ok.sum() < 3:
                continue
            rho, p, kind = PatientMethods._perm_spearman_p(v[ok], s[ok])
            res.append({"region": rg, "rho": rho, "p": p, "test": kind,
                        "n_patients": int(ok.sum())})
        out = pd.DataFrame(res)

        pm = prof[prof["region"] == focus_region].groupby("patient")["mean_meth"].mean()
        sc = score.reindex(pm.index)
        keep = pm.notna() & sc.notna()
        pm, sc = pm[keep], sc[keep]
        rng_ = (prof[prof["region"] == focus_region]
                .groupby("patient")["mean_meth"].agg(["min", "max", "count"]))

        if tier_labels is None:
            tier_labels = {0.0: "0\nneither",
                           1.0: "1\ndeath or\nmetastasis",
                           2.0: "2\ndeath and\nmetastasis"}

        # Second colour is BLACK, not grey: this figure is printed on a poster
        # and mid-grey at 3 pt reproduces faint enough to read as absent.
        ACC, GREY = "#d98416", "#000000"
        with plt.rc_context({"font.size": 11, "axes.labelsize": 12,
                             "xtick.labelsize": 10, "ytick.labelsize": 11}):
            fig, (axA, axB) = plt.subplots(
                1, 2, figsize=(10.2, 4.3),
                gridspec_kw={"width_ratios": [1.55, 1.0]})

            # spread patients within a tier so labels never stack
            off = {}
            for t in sorted(sc.unique()):
                mem = list(pm[sc == t].sort_values().index)
                n = len(mem)
                for k, p_ in enumerate(mem):
                    off[p_] = 0.0 if n == 1 else (k - (n - 1) / 2) * 0.30
            top = float(pm.max())
            for p_ in pm.index:
                x = float(sc[p_]) + off[p_]
                y = float(pm[p_])
                k = int(rng_.loc[p_, "count"])
                if k > 1:
                    lo, hi = rng_.loc[p_, "min"], rng_.loc[p_, "max"]
                    axA.vlines(x, lo, hi, color=GREY, lw=1.8, zorder=2)
                    for yv in (lo, hi):
                        axA.plot([x - .05, x + .05], [yv] * 2, color=GREY,
                                 lw=1.4, zorder=2)
                    top = max(top, hi)
                    axA.text(x, hi + top * .03, f"(n={k})", fontsize=9,
                             color=GREY, ha="center", va="bottom")
                axA.scatter(x, y, s=120, color=ACC, edgecolor="white", lw=1.4,
                            zorder=4)
                axA.annotate(str(p_), (x, y), xytext=(0, -15),
                             textcoords="offset points", fontsize=9.5,
                             color=GREY, ha="center", zorder=5)
            for t in sorted(sc.unique()):
                axA.hlines(pm[sc == t].median(), t - .42, t + .42,
                           color=GREY, lw=2.2, zorder=3)
            ticks = sorted(sc.unique())
            axA.set_xticks(ticks)
            axA.set_xticklabels([tier_labels.get(t, f"{t:g}") for t in ticks])
            axA.set_xlim(min(ticks) - .62, max(ticks) + .62)
            axA.set_ylim(0, top * 1.20)
            axA.set_xlabel("aggressiveness score   (1 point each)")
            axA.set_ylabel(f"{focus_region} methylation (%)")
            axA.set_title(f"{focus_region} methylation rises with aggressiveness",
                          fontsize=12.5, pad=26)
            r0 = out[out["region"] == focus_region].iloc[0]
            axA.text(.02, .975,
                     f"Spearman $\\rho$ = {r0['rho']:+.2f}\n"
                     f"{r0['test']} p = {r0['p']:.3f}   "
                     f"n = {int(r0['n_patients'])} patients",
                     transform=axA.transAxes, va="top", ha="left",
                     fontsize=10.5,
                     bbox=dict(facecolor="#fdf3e3", edgecolor=ACC, lw=1.0, pad=5))
            axA.spines[["top", "right"]].set_visible(False)

            P_COL = 1.62          # x of the p-value column, inside xlim below
            ys = np.arange(len(out))[::-1]
            for y_, (_, row) in zip(ys, out.iterrows()):
                c = ACC if row["region"] == focus_region else GREY
                axB.plot([0, row["rho"]], [y_, y_], color=c, lw=3.0,
                         solid_capstyle="round", zorder=2)
                axB.scatter(row["rho"], y_, s=95, color=c, edgecolor="white",
                            lw=1.3, zorder=3)
                # p values sit in a fixed column at the right edge rather than
                # floating beside each bar. Anchored to rho they land on top of
                # the y tick label whenever rho is near zero, which is exactly
                # the case for the regions that are supposed to read as null.
                axB.text(P_COL, y_, f"p = {row['p']:.3f}",
                         va="center", ha="right", fontsize=10,
                         color=ACC if row["region"] == focus_region else GREY)
            axB.axvline(0, color=GREY, lw=1.0)
            axB.set_yticks(ys)
            axB.set_yticklabels(out["region"], fontsize=11)
            axB.set_xlim(-1.15, P_COL + .05)
            axB.set_ylim(-.6, len(out) - .4)
            axB.set_xticks([-1, -.5, 0, .5, 1])
            axB.set_xlabel("Spearman $\\rho$ vs aggressiveness")
            axB.set_title(f"the signal is specific to {focus_region}",
                          fontsize=12.5, pad=26)
            axB.spines[["top", "right", "left"]].set_visible(False)
            axB.tick_params(axis="y", length=0)

            gname = str(prof["gene"].iloc[0]) if "gene" in prof.columns else ""
            fig.suptitle(title or
                         f"{gname} {focus_region} methylation and clinical "
                         f"aggressiveness",
                         fontsize=13.5, y=1.005)
            fig.tight_layout()

            if actual_dir:
                os.makedirs(actual_dir, exist_ok=True)
                fpath = os.path.join(
                    actual_dir, filename or
                    f"{str(focus_region).lower().replace(' ', '_')}"
                    f"_aggressiveness.pdf")
                fig.savefig(fpath, bbox_inches="tight")
                print(f"Saved: {fpath}")
                plt.close(fig)
            else:
                plt.show()
        return out

    @staticmethod
    def plot_variant_swimmer_cohort(
        samples: list,
        meta: pd.DataFrame,
        surv: pd.DataFrame,
        snv_all: "pd.DataFrame | None" = None,
        sv_all: "pd.DataFrame | None" = None,
        cnv_all: "pd.DataFrame | None" = None,
        kinds: "tuple" = ("SNV", "SV", "CNV"),
        n_groups: int = 5,
        sort_by: str = "os",
        row_height: float = 1.0,
        cnv_gain: float = 2.5,
        cnv_loss: float = 1.5,
        cnv_baseline: str = "ploidy",
        save_dir: "str | None" = None,
        filename_prefix: str = "cohort_swimmer",
        **kwargs,
    ) -> pd.DataFrame:
        """
        plot_variant_swimmer for the whole cohort, split across `n_groups`
        figures per caller — 47 rows will not fit on one page legibly.

        The split must not change what the figures say, so `max_months` and
        `cn_max` are computed ONCE over every sample and handed to each group.
        Letting each group auto-scale is the obvious implementation and it
        quietly makes the figures incomparable: a 40-month bar in group 1 would
        be drawn longer than a 40-month bar in group 5.

        `sort_by="os"` orders rows by overall survival, longest first, so the
        stack of timelines reads as a survival figure; "given" keeps the order
        it was handed (use that to match the column order in the oncoprints).

        Rows are SAMPLES, not patients — a patient with several samples gets a
        row per genome and its timeline repeats, which the p<N>-s<k> labels
        make visible. Collapsing to one row per patient would mean dropping
        genomes.

        Returns one table of everything drawn, with a `group` column.
        """
        sel = [s for s in samples if s in set(meta.index)]
        if not sel:
            print("[WARN] None of the requested samples are in the metadata.")
            return pd.DataFrame()

        surv_of = PatientMethods._surv_index(meta, surv, sel)
        if sort_by == "os":
            def _os(lab):
                r = surv_of.get(lab)
                v = None if r is None else r.get("os_months")
                return -float(v) if pd.notna(v) else float("inf")   # NaN last
            sel = sorted(sel, key=_os)
        elif sort_by != "given":
            raise ValueError('sort_by must be "os" or "given"')

        # one scale for every group
        os_vals = [r.get("os_months") for r in surv_of.values() if r is not None]
        os_vals = [v for v in os_vals if pd.notna(v)]
        max_months = float(max(os_vals)) * 1.14 if os_vals else 120.0

        cn_max = None
        if cnv_all is not None and not cnv_all.empty and "cn" in cnv_all.columns:
            v = cnv_all.loc[cnv_all["sample"].isin(sel), "cn"].dropna()
            cn_max = (float(np.clip(np.ceil(np.percentile(v, 99.5)), 4.0, 12.0))
                      if len(v) else 6.0)

        groups = [list(g) for g in np.array_split(np.array(sel, dtype=object),
                                                  max(1, n_groups)) if len(g)]
        table_of = {"SNV": snv_all, "SV": sv_all, "CNV": cnv_all}
        arg_of = {"SNV": "snv_all", "SV": "sv_all", "CNV": "cnv_all"}

        out = []
        for kind in kinds:
            if table_of.get(kind) is None:
                print(f"[INFO] no {kind} table supplied — skipped")
                continue
            for gi, grp in enumerate(groups, start=1):
                t = PatientMethods.plot_variant_swimmer(
                    grp, meta, surv, kind=kind,
                    cnv_gain=cnv_gain, cnv_loss=cnv_loss,
                    cnv_baseline=cnv_baseline,
                    cn_max=cn_max, max_months=max_months,
                    row_height=row_height,
                    title=f"{kind} landscape and clinical course — "
                          f"group {gi} of {len(groups)} (n={len(grp)})",
                    filename=f"{filename_prefix}_{kind.lower()}_g{gi}.pdf",
                    save_dir=save_dir,
                    **{arg_of[kind]: table_of[kind]}, **kwargs)
                if not t.empty:
                    out.append(t.assign(kind=kind, group=gi))

        return (pd.concat(out, ignore_index=True) if out else pd.DataFrame())
