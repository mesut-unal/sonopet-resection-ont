# toronto_projection

Methylation-based projection of ONT meningioma samples (n=14; 7 patients ×
Sonopet/Resection) onto the Toronto EPIC reference cohort
(Nassiri et al. 2021, *Nature*, n=121).

## Setup

Use the conda environment from the repository root:

```bash
conda env create -f ../environment.yml
conda activate sonopet
```

## Folder layout

```
toronto_projection/
├── toronto_direct_mapping.py        # step 1: Toronto beta matrix → top_probes.pkl
├── build_ont_matrix.py              # step 2: ONT bedMethyl → ont_matrix.pkl
├── toronto_projection.py            # step 3: projection + paper figures
├── input_data/                      # raw inputs (see below)
├── data/                            # small tracked reference files
├── cache_toronto_direct/            # created by step 1
├── cache_toronto_direct_7patients/  # created by step 2, read by step 3
└── output/                          # created by steps 1 and 3
```

## Input data

### `input_data/`

| File | Tracked | Source |
|---|---|---|
| `mg_sample_labels.tsv` | yes | Toronto 4-MG labels (Nassiri et al. 2021) |
| `EPIC_hg38_manifest.tsv.gz` | no | [Zhou lab EPIC hg38 manifest](https://zwdzwd.github.io/InfiniumAnnotation/20180909/EPIC/EPIC.hg38.manifest.tsv.gz) |
| `GSE180061_Matrix_processed.txt.gz` | no (large) | GEO [GSE180061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180061) supplementary file |
| `cpgIslandExt.txt.gz` | no | [UCSC hg38 CpG islands](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz); only needed if `data/probe_to_island.pkl` is missing |

Download the untracked files into `input_data/`:

```bash
cd input_data
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180061/suppl/GSE180061_Matrix_processed.txt.gz
wget -O EPIC_hg38_manifest.tsv.gz https://zwdzwd.github.io/InfiniumAnnotation/20180909/EPIC/EPIC.hg38.manifest.tsv.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz
```

### `data/`

- `consensus_labels_k6.pkl`: Toronto sample cluster labels (k=6, canonical)
- `consensus_matrix_k6.pkl`: 121×121 consensus co-clustering matrix
- `probe_to_island.pkl`: cached EPIC probe → CpG island map
- `probe_to_chrom.pkl`: cached EPIC probe → chromosome map

### ONT methylation (not distributed)

ONT bedMethyl files are unpublished and are not part of this repository.
`toronto_direct_mapping.py` and `build_ont_matrix.py` expect the modkit
sample folders under the directory set in `OUT_BASE`
(placeholder: `/path/to/modkit_output`):

```
/path/to/modkit_output/ONTWGS9-<key>-*/3_methylation/ONTWGS9-<key>-*_CpG_5mC.bed
```

Edit `OUT_BASE` in both scripts before running them.

## Usage

```bash
python toronto_direct_mapping.py   # writes cache_toronto_direct/top_probes.pkl
python build_ont_matrix.py         # writes cache_toronto_direct_7patients/{top_probes,ont_matrix}.pkl
python toronto_projection.py       # writes output/toronto_projection_7patients_<timestamp>/
```

Only steps 1–3 of `toronto_direct_mapping.py` (manifest, beta matrix, top 10,000
probes by MAD) feed the projection; its Random Forest, UMAP and concordance
outputs are exploratory.

## Parameter combinations

Four combinations are run (2 normalization × 2 feature space):

| Tag | Normalization | Features |
|---|---|---|
| `quantile_probe` | Quantile (within-platform) | 7,462 autosomal CpG probes |
| `quantile_cpg_island_rank` | Quantile | 799 CpG islands, rank-transformed |
| `mean_center_probe` | Mean centering | 7,462 autosomal CpG probes |
| `mean_center_cpg_island_rank` | Mean centering | 799 CpG islands, rank-transformed |

Figures used in the manuscript (from `quantile_cpg_island_rank`):

- `mg4_probability_heatmap_quantile_cpg_island_rank.pdf`
- `combined_consensus_quantile_cpg_island_rank.pdf`

See manuscript Methods for full pipeline description.

## Reference

Nassiri F. et al. (2021). A clinically applicable integrative molecular
classification of meningiomas. *Nature* 597, 119–125. GEO accession: GSE180061.
