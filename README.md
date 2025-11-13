# Popformer

An axial attention transformer for haplotype matrices. Self-supervised pre-training
on masked haplotype reconstruction. Evaluated downstream on genotype imputation,
classification for selection/neutral windows, and classification for real/simulated
windows. Also can probably do demographic inference, per-SNP tasks like site classification,
and per-SNP per-haplotype tasks like local ancestry inference.

## Setup

TODO pip install

## Repository

TODO huggingface

- code in `popformer`
- models in `models`
  - models are hf PreTrainedModels
- analysis in `analysis`
- datasets in `datasets`
  - datasets are generally hf Datasets
- figures in `figs`

## TODOs

An overview of what is done here.

- Pre-training models gets us
  - [x] Embedding figures
  - [x] Genotype imputation results
- Fine-tuning on selection gets us
  - [ ] Test split results (classification metrics)
  - [ ] Simulated sweep detection results (detection metrics of some sort ???)
    - [ ] GHIST and
    - [ ] our own
  - [ ] Real sweep detection results
    - [ ] permutation tests on grossman / reich regions, although the reich regions may be a null result
- Linear probes on selection get us
  - [ ] Comparisons between linear probes and fine-tuning
- We should also add
  - [ ] Comparisons with other models (maybe, one summary stat and one)
- We can additionally add
  - [ ] Real/sim classification models