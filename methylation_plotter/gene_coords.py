"""
gene_coords.py
==============
Genomic region configuration for methylation browser plots.
Import in plotting scripts as:
    from gene_coords import GENE_COORDS

Provenance (all values re-derived and verified 2026-08-27)
----------------------------------------------------------
hg38 : full gene span from GENCODE v44 `gene` features, taken from
       BTC_ONT/decoil-viz/gencode.v44.primary_assembly.basic.annotation.gtf.gz
       — the same annotation the rest of this pipeline uses, so it is
       reproducible locally. Spot-checked against the UCSC hg38 HGNC track and
       Ensembl GRCh38; identical in GENCODE v42 and v44.
hg19 : UCSC hg19 HGNC track (api.genome.ucsc.edu/search?genome=hg19).

These are **full gene spans**, i.e. the union of all annotated transcripts —
not the MANE/canonical transcript span, which is shorter for several genes
(PTEN, TP53, PBRM1, LYST, SETD2, NF1, SMARCB1 among them). Use a gene span for
"does anything overlap this gene" questions; use a transcript span if you
specifically mean the canonical isoform.

The previous version of this table was unreliable and has been replaced
wholesale. Several entries were wrong by hundreds of kb: NF2's "hg38" value was
actually its hg19 coordinate, SMO/KLF4/TRAF7 were off by 68–357 kb, and
SMARCE1's hg38 value (38,617,478) matched neither build — the real hg38 start
is 40,624,962, a ~2 Mb error. The old file is preserved as
gene_coords.py.bak_precoordfix.

Coordinates are 1-based inclusive (GTF/HGNC convention), matching how
methylation_browser.py slices regions.
"""

GENE_COORDS = {

    # ── Chromosome 22 ──────────────────────────────────────────────────────────
    "NF2": {
        "hg38": {"chr": "22", "start": 29_603_556,  "end": 29_698_598},
        "hg19": {"chr": "22", "start": 29_999_542,  "end": 30_094_587},
    },
    "SMARCB1": {
        "hg38": {"chr": "22", "start": 23_786_931,  "end": 23_838_009},
        "hg19": {"chr": "22", "start": 24_129_118,  "end": 24_180_196},
    },

    # ── Chromosome 16 ──────────────────────────────────────────────────────────
    "TRAF7": {
        "hg38": {"chr": "16", "start":  2_155_698,  "end":  2_178_129},
        "hg19": {"chr": "16", "start":  2_205_699,  "end":  2_228_130},
    },

    # ── Chromosome 9 ───────────────────────────────────────────────────────────
    "KLF4": {
        "hg38": {"chr": "9",  "start": 107_484_852, "end": 107_490_482},
        "hg19": {"chr": "9",  "start": 110_247_133, "end": 110_252_763},
    },
    "CDKN2A": {
        "hg38": {"chr": "9",  "start":  21_967_752, "end":  21_995_301},
        "hg19": {"chr": "9",  "start":  21_967_751, "end":  21_995_300},
    },
    "CDKN2B": {
        "hg38": {"chr": "9",  "start":  22_002_903, "end":  22_009_305},
        "hg19": {"chr": "9",  "start":  22_002_902, "end":  22_009_304},
    },
    # composite 9p21 window: CDKN2A start -> CDKN2B end (the two genes sit
    # 7.6 kb apart and are almost always co-deleted)
    "CDKN2A/B": {
        "hg38": {"chr": "9",  "start":  21_967_752, "end":  22_009_305},
        "hg19": {"chr": "9",  "start":  21_967_751, "end":  22_009_304},
    },

    # ── Chromosome 14 ──────────────────────────────────────────────────────────
    "AKT1": {
        "hg38": {"chr": "14", "start": 104_769_349, "end": 104_795_751},
        "hg19": {"chr": "14", "start": 105_235_686, "end": 105_262_096},
    },

    # ── Chromosome 7 ───────────────────────────────────────────────────────────
    "SMO": {
        "hg38": {"chr": "7",  "start": 129_188_633, "end": 129_213_545},
        "hg19": {"chr": "7",  "start": 128_828_474, "end": 128_853_386},
    },
    "EGFR": {
        "hg38": {"chr": "7",  "start":  55_019_017, "end":  55_211_628},
        "hg19": {"chr": "7",  "start":  55_086_710, "end":  55_279_321},
    },

    # ── Chromosome 3 ───────────────────────────────────────────────────────────
    "PIK3CA": {
        "hg38": {"chr": "3",  "start": 179_148_114, "end": 179_240_093},
        "hg19": {"chr": "3",  "start": 178_865_902, "end": 178_957_881},
    },
    "BAP1": {
        "hg38": {"chr": "3",  "start":  52_401_008, "end":  52_410_008},
        "hg19": {"chr": "3",  "start":  52_435_024, "end":  52_444_024},
    },

    # ── Chromosome 17 ──────────────────────────────────────────────────────────
    "POLR2A": {
        "hg38": {"chr": "17", "start":   7_484_366, "end":   7_514_616},
        "hg19": {"chr": "17", "start":   7_387_685, "end":   7_417_933},
    },
    "TP53": {
        "hg38": {"chr": "17", "start":   7_661_779, "end":   7_687_538},
        "hg19": {"chr": "17", "start":   7_565_097, "end":   7_590_864},
    },
    "SMARCE1": {
        "hg38": {"chr": "17", "start":  40_624_962, "end":  40_648_654},
        "hg19": {"chr": "17", "start":  38_781_214, "end":  38_804_906},
    },

    # ── Chromosome 5 ───────────────────────────────────────────────────────────
    "TERT": {
        "hg38": {"chr": "5",  "start":   1_253_147, "end":   1_295_068},
        "hg19": {"chr": "5",  "start":   1_253_262, "end":   1_295_183},
    },

    # ── Chromosome 10 ──────────────────────────────────────────────────────────
    "PTEN": {
        "hg38": {"chr": "10", "start":  87_862_638, "end":  87_971_930},
        "hg19": {"chr": "10", "start":  89_622_395, "end":  89_731_687},
    },

    # ── Chordoma panel additions ───────────────────────────────────────────────
    # Same provenance as everything above: hg38 from GENCODE v44, hg19 from the
    # UCSC hg19 HGNC track. Gene length is identical in both builds for all of
    # these, which is the basic sanity check on a liftover.
    "TBXT": {   # brachyury — the chordoma lineage marker
        "hg38": {"chr": "6",  "start": 166_157_656, "end": 166_168_700},
        "hg19": {"chr": "6",  "start": 166_571_144, "end": 166_582_188},
    },
    "FN1": {
        "hg38": {"chr": "2",  "start": 215_360_440, "end": 215_436_073},
        "hg19": {"chr": "2",  "start": 216_225_163, "end": 216_300_796},
    },
    "SOX9": {
        "hg38": {"chr": "17", "start":  72_121_020, "end":  72_126_416},
        "hg19": {"chr": "17", "start":  70_117_161, "end":  70_122_557},
    },
    "NF1": {
        "hg38": {"chr": "17", "start":  31_094_927, "end":  31_382_116},
        "hg19": {"chr": "17", "start":  29_421_945, "end":  29_709_134},
    },
    "PBRM1": {
        "hg38": {"chr": "3",  "start":  52_545_352, "end":  52_685_917},
        "hg19": {"chr": "3",  "start":  52_579_368, "end":  52_719_933},
    },
    "SETD2": {
        "hg38": {"chr": "3",  "start":  47_016_428, "end":  47_164_113},
        "hg19": {"chr": "3",  "start":  47_057_918, "end":  47_205_603},
    },
    "LYST": {
        "hg38": {"chr": "1",  "start": 235_661_041, "end": 235_883_724},
        "hg19": {"chr": "1",  "start": 235_824_341, "end": 236_047_024},
    },
}

# The chordoma oncoprint panel, in oncoprint row order. Every name resolves in
# GENE_COORDS above.
CHORDOMA_PANEL = [
    "TBXT", "CDKN2A", "CDKN2B", "PIK3CA", "PTEN", "PBRM1", "SETD2",
    "LYST", "TP53", "NF1", "SMARCB1", "FN1", "SOX9", "EGFR",
]


def hg38(gene):
    """{'chrom','start','end'} for one gene — the flat shape variations_analyzer uses."""
    e = GENE_COORDS[gene]["hg38"]
    return {"chrom": str(e["chr"]), "start": int(e["start"]), "end": int(e["end"])}


def check_against_gtf(gtf_path, build="hg38", verbose=True):
    """
    Re-derive the hg38 spans from a GENCODE GTF and report any drift.

    Run this after changing annotation release:
        python -c "import gene_coords as g; g.check_against_gtf('/path/gencode.gtf.gz')"

    Returns {gene: (table_span, gtf_span)} for entries that disagree. hg19 is
    not checked — it comes from the UCSC hg19 HGNC track, not from this GTF.
    """
    import gzip
    import re

    if build != "hg38":
        raise ValueError("only hg38 can be checked against a GRCh38 GENCODE GTF")

    name_re = re.compile(r'gene_name "([^"]+)"')
    type_re = re.compile(r'gene_type "([^"]+)"')
    want = {g for g in GENE_COORDS if "/" not in g}
    found, found_type = {}, {}

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "gene" or "_" in f[0]:
                continue
            m = name_re.search(f[8])
            if not m or m.group(1) not in want:
                continue
            gtype = type_re.search(f[8])
            gtype = gtype.group(1) if gtype else "unknown"
            name = m.group(1)
            if name in found and found_type[name] == "protein_coding" \
                    and gtype != "protein_coding":
                continue
            found[name] = (f[0].replace("chr", ""), int(f[3]), int(f[4]))
            found_type[name] = gtype

    bad = {}
    for gene in sorted(want):
        have = GENE_COORDS[gene][build]
        mine = (str(have["chr"]), have["start"], have["end"])
        theirs = found.get(gene)
        if theirs is None:
            if verbose:
                print(f"  {gene:9s} not in GTF")
            continue
        if mine != theirs:
            bad[gene] = (mine, theirs)
            if verbose:
                print(f"  {gene:9s} table chr{mine[0]}:{mine[1]:,}-{mine[2]:,}"
                      f"  GTF chr{theirs[0]}:{theirs[1]:,}-{theirs[2]:,}")
    if verbose:
        print(f"{len(bad)} of {len(want)} entries differ from {gtf_path}")
    return bad
