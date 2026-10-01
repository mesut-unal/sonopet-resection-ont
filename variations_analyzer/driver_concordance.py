"""
driver_concordance.py

Driver-panel status of SNVs/indels and copy number across samples, with optional
comparison between conditions within a group (e.g. R vs S per patient).

Samples are organized as
    groups = {group_label: {condition_label: sample_name}}
e.g. {"P1": {"R": "ONTWGS9-2-...", "S": "ONTWGS9-1-..."}}.
With one condition per group (e.g. {"S01": {"T": "..."}}) the module reports
per-sample driver status without a comparison.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import pysam


PLOT_CFG = {
    "fontsize_title": 20,
    "fontsize_label": 15,
    "fontsize_tick": 15,
    "marker_size": 80,
}


def _save_fig(fig, path, fmt="pdf"):
    fig.savefig(f"{path}.{fmt}", bbox_inches="tight")


class SonopetMethods:

    CONC_COLORS = {
        "all_called":          "#1b7837",
        "detected_not_called": "#a6dba0",
        "missing":             "#762a83",
        "low_coverage":        "#bdbdbd",
    }

    # -----------------------------------------------------------------
    # Grouping
    # -----------------------------------------------------------------

    @staticmethod
    def build_groups(sample_names, id_to_label,
                     parse_id=lambda s: int(str(s).split("-")[1]),
                     split_label=lambda l: (f"P{l[:-1]}", l[-1]),
                     require=None):
        """
        sample_names : iterable of sample names
        id_to_label  : {sample_id: label}, e.g. SEQID_TO_SR
        parse_id     : sample_name -> sample_id
        split_label  : label -> (group, condition)
        require      : conditions a group must have to be kept, e.g. ("R", "S")
        Returns {group: {condition: sample_name}}.
        """
        id_to_sample = {}
        for s in sample_names:
            try:
                id_to_sample[parse_id(s)] = s
            except (IndexError, ValueError):
                pass
        groups = defaultdict(dict)
        for sid, label in id_to_label.items():
            if sid in id_to_sample:
                g, c = split_label(label)
                groups[g][c] = id_to_sample[sid]
        out = {}
        for g, conds in groups.items():
            if require and not set(require) <= set(conds):
                print(f"[INFO] {g}: missing {set(require) - set(conds)}, skipping.")
                continue
            out[g] = conds
        return dict(sorted(out.items()))

    @staticmethod
    def _conditions(groups):
        return list(dict.fromkeys(c for conds in groups.values() for c in conds))

    # -----------------------------------------------------------------
    # Panel
    # -----------------------------------------------------------------

    @staticmethod
    def load_driver_panel(path):
        """
        TSV columns: name, gene, chrom, start, end, kind
        kind: gene | hotspot | arm   (1-based, inclusive, no flank)
        Rows with missing coordinates are dropped with a warning.
        """
        panel = pd.read_csv(path, sep="\t", comment="#", dtype={"chrom": str})
        missing = panel[["start", "end"]].isna().any(axis=1)
        if missing.any():
            print(f"[WARN] No coordinates, dropped: {', '.join(panel.loc[missing, 'name'])}")
        panel = panel.loc[~missing].copy()
        panel[["start", "end"]] = panel[["start", "end"]].astype(int)
        return panel.reset_index(drop=True)

    @staticmethod
    def _annotate_panel(df, panel, flank=0):
        """Keep variants inside panel genes/hotspots; flank applies to genes only."""
        parts = []
        for _, row in panel[panel["kind"].isin(["gene", "hotspot"])].iterrows():
            pad = 0 if row["kind"] == "hotspot" else flank
            m = (df["chrom"] == row["chrom"]) & df["pos"].between(row["start"] - pad, row["end"] + pad)
            if m.any():
                parts.append(df.loc[m].assign(driver=row["name"], gene=row["gene"],
                                              driver_kind=row["kind"]))
        if not parts:
            return df.iloc[0:0].assign(driver="", gene="", driver_kind="")
        hits = pd.concat(parts, ignore_index=True)
        # a variant in both a gene and a hotspot row keeps the hotspot label
        hits = hits.sort_values("driver_kind", key=lambda s: s.ne("hotspot"), kind="stable")
        return hits.drop_duplicates(subset=["sample", "key"], keep="first")

    # -----------------------------------------------------------------
    # Read-level check
    # -----------------------------------------------------------------

    @staticmethod
    def _bam_contig(bam, chrom):
        """Match a panel chromosome name to the BAM's contig naming (chr-prefixed or not)."""
        refs = set(bam.references)
        for c in (str(chrom), f"chr{chrom}", str(chrom).replace("chr", "")):
            if c in refs:
                return c
        return None

    @staticmethod
    def _force_count(bam, chrom, pos, ref, alt, min_mq=20, min_bq=10):
        """Depth and alt-supporting reads at a VCF position (SNV by base, indel by length)."""
        chrom = SonopetMethods._bam_contig(bam, chrom)
        if chrom is None:
            return np.nan, np.nan
        is_snv = len(ref) == 1 and len(alt) == 1
        indel_len = len(alt) - len(ref)
        dp = n_alt = 0
        for col in bam.pileup(chrom, pos - 1, pos, truncate=True,
                              min_mapping_quality=min_mq, min_base_quality=min_bq):
            for pr in col.pileups:
                dp += 1
                if is_snv:
                    if pr.query_position is not None and \
                            pr.alignment.query_sequence[pr.query_position] == alt:
                        n_alt += 1
                elif pr.indel == indel_len:
                    n_alt += 1
        return dp, n_alt

    @staticmethod
    def _side_state(status, fdp, falt, min_alt, min_dp):
        if status == "PASS":
            return "called"
        if status == "filtered":
            return "detected"
        if pd.isna(fdp):
            return "not_detected"
        if fdp < min_dp:
            return "low_cov"
        return "detected" if falt >= min_alt else "not_detected"

    # -----------------------------------------------------------------
    # SNV / indel driver table
    # -----------------------------------------------------------------

    @staticmethod
    def driver_concordance_table(variants, panel, groups, bam_by_sample=None, flank=0,
                                 min_alt=2, min_dp=10, min_mq=20, min_bq=10):
        """
        variants      : DataFrame with sample, chrom, pos, ref, alt, key, filter, vaf, dp
                        (e.g. clairs_all_raw)
        groups        : {group: {condition: sample_name}}
        bam_by_sample : {sample_name: bam_path}; if given, conditions where a variant
                        is not PASS are re-checked by read counting.

        Every panel variant PASS in at least one sample of a group is reported with
        its status in all samples of that group. Returns (table, summary).
        """
        conditions = SonopetMethods._conditions(groups)
        hits = SonopetMethods._annotate_panel(variants, panel, flank)

        rows = []
        for group, conds in groups.items():
            sub = hits[hits["sample"].isin(conds.values())]
            called = sub.loc[sub["filter"] == "PASS", "key"].unique()
            if len(called) == 0:
                continue
            bams = {}
            if bam_by_sample:
                bams = {c: pysam.AlignmentFile(bam_by_sample[s]) for c, s in conds.items()}

            for k in called:
                first = sub[sub["key"] == k].iloc[0]
                row = {"group": group, "gene": first["gene"], "driver": first["driver"],
                       "chrom": first["chrom"], "pos": first["pos"],
                       "ref": first["ref"], "alt": first["alt"], "key": k}
                states = set()
                for c, s in conds.items():
                    rec = sub[(sub["sample"] == s) & (sub["key"] == k)]
                    if rec.empty:
                        status, vaf, dp = "absent", np.nan, np.nan
                    else:
                        status = "PASS" if rec.iloc[0]["filter"] == "PASS" else "filtered"
                        vaf, dp = rec.iloc[0]["vaf"], rec.iloc[0]["dp"]
                    fdp = falt = np.nan
                    if bams:
                        fdp, falt = SonopetMethods._force_count(
                            bams[c], row["chrom"], int(row["pos"]), row["ref"], row["alt"],
                            min_mq=min_mq, min_bq=min_bq)
                    fvaf = falt / fdp if fdp > 0 else np.nan
                    state = SonopetMethods._side_state(status, fdp, falt, min_alt, min_dp)
                    states.add(state)
                    row.update({
                        f"sample_{c}": s,
                        f"status_{c}": status, f"state_{c}": state,
                        f"vaf_{c}": vaf, f"dp_{c}": dp,
                        f"fdp_{c}": fdp, f"falt_{c}": falt, f"fvaf_{c}": fvaf,
                        f"dispvaf_{c}": vaf if status != "absent" else fvaf,
                    })
                if states == {"called"}:
                    row["concordance"] = "all_called"
                elif "low_cov" in states:
                    row["concordance"] = "low_coverage"
                elif "not_detected" in states:
                    row["concordance"] = "missing"
                else:
                    row["concordance"] = "detected_not_called"
                rows.append(row)

            for b in bams.values():
                b.close()

        table = pd.DataFrame(rows)
        cats = list(SonopetMethods.CONC_COLORS)
        all_groups = list(groups)
        if table.empty:
            summary = pd.DataFrame(0, index=all_groups, columns=cats)
        else:
            summary = (table.groupby(["group", "concordance"]).size()
                       .unstack(fill_value=0)
                       .reindex(index=all_groups, columns=cats, fill_value=0))
        summary.index.name = "group"
        evaluable = summary[cats].drop(columns="low_coverage").sum(axis=1)
        summary["n_variants"] = summary[cats].sum(axis=1)
        if len(conditions) > 1:
            summary["concordance_rate"] = (
                (summary["all_called"] + summary["detected_not_called"])
                / evaluable.replace(0, np.nan))
        return table, summary

    # -----------------------------------------------------------------
    # Copy number driver table
    # -----------------------------------------------------------------

    @staticmethod
    def _cn_call(l2, homdel_thr, loss_thr, gain_thr):
        if pd.isna(l2):
            return "NA"
        if l2 <= homdel_thr:
            return "HD"
        if l2 <= loss_thr:
            return "loss"
        if l2 >= gain_thr:
            return "gain"
        return "neutral"

    @staticmethod
    def driver_cnv_table(cns_by_sample, panel, groups,
                         homdel_thr=-1.1, loss_thr=-0.25, gain_thr=0.2):
        """
        cns_by_sample : {sample_name: CNVkit .cns DataFrame (chromosome, start, end, log2)}
        Length-weighted mean log2 over each panel gene/arm, called per sample.
        Thresholds are not purity-adjusted.
        """
        regions = panel[panel["kind"].isin(["gene", "arm"])]
        rows = []
        for group, conds in groups.items():
            for _, reg in regions.iterrows():
                row = {"group": group, "region": reg["name"], "kind": reg["kind"]}
                for c, s in conds.items():
                    cns = cns_by_sample.get(s)
                    l2 = np.nan
                    if cns is not None:
                        seg = cns[(cns["chromosome"] == reg["chrom"]) &
                                  (cns["end"] > reg["start"]) & (cns["start"] < reg["end"])]
                        if not seg.empty:
                            ov = (np.minimum(seg["end"], reg["end"]) -
                                  np.maximum(seg["start"], reg["start"])).clip(lower=1)
                            l2 = np.average(seg["log2"], weights=ov)
                    row[f"log2_{c}"] = l2
                    row[f"call_{c}"] = SonopetMethods._cn_call(l2, homdel_thr, loss_thr, gain_thr)
                if len(conds) > 1:
                    row["concordant"] = len({row[f"call_{c}"] for c in conds}) == 1
                rows.append(row)
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    @staticmethod
    def _group_columns(ax, cols, cond_colors, plot_cfg, multi):
        """x ticks per (group, condition), colored by condition, with group separators."""
        labels = [f"{g}\n{c}" if multi else g for g, c in cols]
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, fontsize=plot_cfg["fontsize_tick"])
        for t, (_, c) in zip(ax.get_xticklabels(), cols):
            t.set_color(cond_colors.get(c, "black"))
        for j in range(1, len(cols)):
            if cols[j][0] != cols[j - 1][0]:
                ax.axvline(j - 0.5, color="white", lw=4)

    @staticmethod
    def plot_driver_concordance(table, summary, panel, groups,
                                conditions=None, cond_colors=None, xy=None, plot_cfg=None,
                                vaf_range=(0.0, 1.0), save_path=None, fmt="pdf"):
        """
        A: gene x sample VAF grid (always)
        B: VAF scatter, only when xy=(x_condition, y_condition) is given
        C: per-group concordance bars, only with more than one condition
        conditions: column order within a group, e.g. ("R", "S")
        vaf_range: (vmin, vmax) for the VAF colorbar and the scatter axes
        """
        plot_cfg = plot_cfg or PLOT_CFG
        cond_colors = cond_colors or {}
        conditions = conditions or SonopetMethods._conditions(groups)
        multi = len(conditions) > 1
        genes = [g for g in dict.fromkeys(panel["gene"]) if g in set(table.get("gene", []))]
        cats = list(SonopetMethods.CONC_COLORS)
        ms = plot_cfg.get("marker_size", 80)
        leg = dict(frameon=True, framealpha=0.7, facecolor="white",
                   fontsize=plot_cfg["fontsize_tick"] - 3)

        panels = ["grid"] + (["scatter"] if xy else []) + (["bar"] if multi else [])
        ratios = {"grid": 2.2, "scatter": 1, "bar": 1}
        fig, axes = plt.subplots(
            1, len(panels), figsize=(8 + 8 * len(panels), max(6, 0.55 * len(genes) + 3)),
            gridspec_kw={"width_ratios": [ratios[p] for p in panels]}, squeeze=False)
        ax = dict(zip(panels, axes[0]))

        # --- grid ---------------------------------------------------
        cols = [(g, c) for g, conds in groups.items() for c in conditions if c in conds]
        M = np.full((len(genes), len(cols)), np.nan)
        ST = np.full(M.shape, "", dtype=object)
        disp_cols = [f"dispvaf_{c}" for c in conditions]
        rank = {"not_detected": 0, "low_cov": 1, "detected": 2, "called": 3}
        for i, gene in enumerate(genes):
            for j, (g, c) in enumerate(cols):
                t = table[(table["gene"] == gene) & (table["group"] == g)]
                if t.empty:
                    continue
                disp = t[[d for d in disp_cols if d in t]].max(axis=1)
                worst = t[[f"state_{k}" for k in conditions if f"state_{k}" in t]] \
                          .apply(lambda r: min(rank[v] for v in r), axis=1)
                pick = pd.DataFrame({"w": worst, "v": disp.fillna(0)}) \
                         .sort_values(["w", "v"], ascending=[True, False]).index[0]
                top = t.loc[pick]
                M[i, j] = top[f"dispvaf_{c}"]
                ST[i, j] = top[f"state_{c}"]

        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("#f0f0f0")
        axA = ax["grid"]
        im = axA.imshow(M, cmap=cmap, vmin=vaf_range[0], vmax=vaf_range[1], aspect="auto")
        plt.rcParams["hatch.linewidth"] = 0.8
        state_style = {
            "called":       dict(hatch=None,   edgecolor="black",   lw=0.8),
            "detected":     dict(hatch="O", edgecolor="black",   lw=0.8),
            "not_detected": dict(hatch="X",  edgecolor="black",   lw=0.8),
            "low_cov":      dict(hatch="///",  edgecolor="#636363", lw=0.8),
        }
        state_label = {
            "called":       "PASS call",
            "detected":     "Failed filters, reads present",
            "not_detected": "No supporting reads",
            "low_cov":      "Depth too low to judge",
        }
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                st = ST[i, j]
                if not st:
                    continue
                kw = state_style[st]
                axA.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        hatch=kw["hatch"], edgecolor=kw["edgecolor"],
                                        linewidth=kw["lw"]))
        present = set(ST.ravel())
        handles = [Patch(facecolor="#fdbb84" if st == "called" else "white",
                         hatch=kw["hatch"], edgecolor=kw["edgecolor"],
                         label=state_label[st])
                   for st, kw in state_style.items() if st in present]
        SonopetMethods._group_columns(axA, cols, cond_colors, plot_cfg, multi)
        axA.set_yticks(range(len(genes)))
        axA.set_yticklabels(genes, fontsize=plot_cfg["fontsize_tick"], fontstyle="italic")
        axA.set_title("Driver alterations by sample", fontsize=plot_cfg["fontsize_title"])
        cb = fig.colorbar(im, ax=axA, fraction=0.03, pad=0.01)
        cb.set_label("VAF", fontsize=plot_cfg["fontsize_label"])
        cb.ax.tick_params(labelsize=plot_cfg["fontsize_tick"])
        axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
           ncol=4, **{**leg, "fontsize": 16}, handlelength=3, handleheight=2)

        # --- scatter ------------------------------------------------
        if "scatter" in ax:
            cx, cy = xy
            axB = ax["scatter"]
            for cat in cats:
                t = table[table["concordance"] == cat] if not table.empty else table
                if len(t):
                    axB.scatter(t[f"dispvaf_{cx}"].fillna(vaf_range[0]),
                                t[f"dispvaf_{cy}"].fillna(vaf_range[0]), s=ms,
                                color=SonopetMethods.CONC_COLORS[cat], edgecolor="black",
                                label=f"{cat.replace('_', ' ')} (n={len(t)})")
            lo, hi = vaf_range
            pad = 0.02 * (hi - lo)
            axB.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1)
            axB.set_xlim(lo - pad, hi + pad)
            axB.set_ylim(lo - pad, hi + pad)
            axB.set_xlabel(f"VAF ({cx})", fontsize=plot_cfg["fontsize_label"])
            axB.set_ylabel(f"VAF ({cy})", fontsize=plot_cfg["fontsize_label"])
            axB.tick_params(labelsize=plot_cfg["fontsize_tick"])
            axB.set_title("Driver variant VAF", fontsize=plot_cfg["fontsize_title"])
            axB.set_aspect("equal")
            if not table.empty:
                axB.legend(loc="upper left", **leg)

        # --- bars ---------------------------------------------------
        if "bar" in ax:
            axC = ax["bar"]
            grp = list(summary.index)
            x = np.arange(len(grp))
            bottom = np.zeros(len(grp))
            for cat in cats:
                v = summary[cat].values
                axC.bar(x, v, bottom=bottom, color=SonopetMethods.CONC_COLORS[cat], edgecolor="black")
                bottom += v
            axC.set_xticks(x)
            axC.set_xticklabels(grp, fontsize=plot_cfg["fontsize_tick"])
            axC.tick_params(axis="y", labelsize=plot_cfg["fontsize_tick"])
            axC.set_ylabel("Driver variants", fontsize=plot_cfg["fontsize_label"])
            axC.set_title("Concordance per group", fontsize=plot_cfg["fontsize_title"])
            axC.legend(handles=[Patch(facecolor=SonopetMethods.CONC_COLORS[cat], edgecolor="black",
                                      label=cat.replace("_", " ")) for cat in cats],
                       loc="upper right", **leg)

        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path, fmt)
        return fig

    @staticmethod
    def plot_driver_cnv(cnv_table, groups, conditions=None, cond_colors=None, plot_cfg=None,
                        save_path=None, fmt="pdf"):
        """Region x sample heatmap of log2, annotated with the CN call."""
        if cnv_table.empty:
            print("[WARN] Empty CNV table: no gene/arm rows with coordinates in the panel.")
            return None
        plot_cfg = plot_cfg or PLOT_CFG
        cond_colors = cond_colors or {}
        conditions = conditions or SonopetMethods._conditions(groups)
        regions = list(dict.fromkeys(cnv_table["region"]))
        cols = [(g, c) for g, conds in groups.items() for c in conditions if c in conds]
        M = np.full((len(regions), len(cols)), np.nan)
        C = np.full(M.shape, "", dtype=object)
        short = {"HD": "HD", "loss": "L", "gain": "G", "neutral": "", "NA": ""}
        for i, r in enumerate(regions):
            for j, (g, c) in enumerate(cols):
                t = cnv_table[(cnv_table["region"] == r) & (cnv_table["group"] == g)]
                if len(t):
                    M[i, j] = t.iloc[0][f"log2_{c}"]
                    C[i, j] = short[t.iloc[0][f"call_{c}"]]

        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(cols) + 3), max(5, 0.5 * len(regions) + 2)))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#f0f0f0")
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if C[i, j]:
                    ax.text(j, i, C[i, j], ha="center", va="center",
                            fontsize=plot_cfg["fontsize_tick"] - 2, fontweight="bold")
        SonopetMethods._group_columns(ax, cols, cond_colors, plot_cfg, len(conditions) > 1)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions, fontsize=plot_cfg["fontsize_tick"])
        ax.set_title("Driver copy number by sample", fontsize=plot_cfg["fontsize_title"])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("log2 ratio", fontsize=plot_cfg["fontsize_label"])
        cb.ax.tick_params(labelsize=plot_cfg["fontsize_tick"])
        fig.tight_layout()
        if save_path:
            _save_fig(fig, save_path, fmt)
        return fig