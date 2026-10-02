# sonopet-resection-ont

Analysis code for the Sonopet vs. conventional resection ONT sequencing study
of meningioma (7 patients, paired Sonopet (S) and Resection (R) samples).

## Setup

```bash
conda env create -f environment.yml
conda activate sonopet
```

`mutation_signatures_sonopet.ipynb` also needs the GRCh38 reference genome for
SigProfiler, installed once with:

```python
from SigProfilerMatrixGenerator import install as genInstall
genInstall.install("GRCh38")
```

## Folders

### `variations_analyzer/`
Genomic comparisons between Sonopet and Resection samples.

| File | Content |
|---|---|
| `sonopet_analysis.ipynb` | Sequencing yield (NanoStats), SNVs (ClairS-TO) with VAF filtering, CNVs (CNVkit); exports driver-panel inputs for `driver_concordance.ipynb` |
| `driver_concordance.ipynb` | Driver-panel concordance analysis and plots between paired S/R samples |
| `mutation_signatures_sonopet.ipynb` | SBS mutational signatures (SigProfiler) |
| `nanoplot_replot_sonopet.ipynb` | Read length vs. quality plots from NanoPlot output |
| `wakhan_replot.ipynb` | Ploidy / purity plots from Wakhan output |
| `variations_analyzer.py` | Module with SNV/SV/CNV loading, filtering and plotting methods (`VariationAnalyzer`, `SNVmethods`, `SVmethods`, `CNVmethods`) |
| `driver_concordance.py` | `SonopetMethods` class for driver-panel SNV/indel and copy-number status |
| `driver_panel.tsv` | Meningioma driver gene panel used by the concordance analysis |

Run the notebooks from inside this folder so the two modules can be imported.

### `methylation_plotter/`
Per-patient CpG methylation plots comparing Sonopet and Resection samples,
from modkit bedMethyl files.

| File | Content |
|---|---|
| `methylation_browser.py` | Methylation browser: % methylation and coverage for S and R across a gene, with CpG islands |
| `promoter_methylation_heatmap.py` | Binned heatmap of promoter CpG methylation %, R vs S |
| `gene_coords.py` | hg38 gene coordinates used by both scripts |
| `sample_utils.py` | Sample discovery helpers for `promoter_methylation_heatmap.py` |
| `hg38_promoters_tss2kb.bed` | hg38 RefSeq promoter regions (TSS −2 kb to +500 bp), input for `promoter_methylation_heatmap.py` |

Set the placeholder paths first: `OUT_BASE` in `methylation_browser.py` and
`OUT_ROOT` in `sample_utils.py` (or the `MODKIT_OUT_ROOT` environment
variable). Expected inputs:

```
methylation_browser.py:            OUT_BASE/<sample>/3_methylation/<sample>_CpG_5mC.bed
                                   OUT_BASE/cpgIslandExt_hg38.txt.gz   (UCSC hg38 cpgIslandExt)
promoter_methylation_heatmap.py:   OUT_ROOT/<cohort>/modkit/<sample>/3_methylation/<sample>_CpG_5mC.bed
                                   methylation_plotter/hg38_promoters_tss2kb.bed   (included)
```

Usage (patients 1 and 7 are shown in the manuscript):

```bash
cd methylation_plotter

# Methylation browser -> plots/browser/patient{1,7}_<GENE>_browser.pdf
python methylation_browser.py --patient 1 --gene <GENE>
python methylation_browser.py --patient 7 --gene <GENE>

# Promoter methylation R vs S -> plots/promoter_heatmap/promoter_density_<R>_vs_<S>.pdf
python promoter_methylation_heatmap.py --cohort ONTWGS9 --pair ONTWGS9-2-238702-NP01 ONTWGS9-1-238701-NP01    # patient 1
python promoter_methylation_heatmap.py --cohort ONTWGS9 --pair ONTWGS9-12-240190-NP01 ONTWGS9-13-240191-NP01  # patient 7
```

The browser draws the gene body only; set `FLANK_BP` in
`methylation_browser.py` to add context on each side.

### `toronto_projection/`
Methylation-based classification of the ONT samples into the Toronto
meningioma molecular groups (MG1–4; Nassiri et al. 2021) by projecting them
onto the 121-sample EPIC reference cohort (GEO GSE180061). Produces the
MG probability heatmap and the combined consensus-clustering figure.
See [toronto_projection/README.md](toronto_projection/README.md).

### Top level
- `environment.yml`: conda environment for all code in this repository.

## Data

Raw ONT sequencing data and derived variant calls are not distributed with
this repository. Input paths in the scripts and notebooks use
`/path/to/...` placeholders and must be set before running.

## License

Code in this repository is released under the [MIT License](LICENSE).
