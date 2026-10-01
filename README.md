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

### `toronto_projection/`
Methylation-based classification of the ONT samples into the Toronto
meningioma molecular groups (MG1–4; Nassiri et al. 2021) by projecting them
onto the 121-sample EPIC reference cohort (GEO GSE180061). Produces the
MG probability heatmap and the combined consensus-clustering figure.
See [toronto_projection/README.md](toronto_projection/README.md).

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

### Top level
- `environment.yml`: conda environment for all code in this repository.

## Data

Raw ONT sequencing data and derived variant calls are not distributed with
this repository. Input paths in the scripts and notebooks use
`/path/to/...` placeholders and must be set before running.
