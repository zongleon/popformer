# analysis

evaluating Popformer and others on various tasks

## experiments

### mlm: `test_mlm.py`
Benchmark Popformer's MLM accuracy against two baselines

### imputation: `test_imp.py`
Benchmark genotype imputation quality and compares model predictions to baselines and IMPUTE5

### embeddings: `test_embeddings.py`
Extracts Popformer hidden-state embeddings, runs PCA, and visualize. 1000 Genomes real data and simulated data.

### simulated selection: `test_selection.py`
Evaluates selection-detection models (Popformer linear probe, Popformer fine-tuned, ResNet34, FASTER-NN, and summary statistics) on simulated datasets. 

### real selection: `test_selection_real.py`
Same thing for 1000 Genomes CEU/CHB/YRI genome scans, using Grossman *et al.* and Akbari *et al.* labelled regions as ground truth.

## training in `train/`
Shell scripts and Python code for the three main training experiments:
1. **Real-data pipeline** — pretrain on 1000G, fine-tune on simulated CEU, evaluate cross-population.
2. **Pretraining effects** — vary bottleneck strength or constant *N* during pretraining to measure transfer.
3. **Data-size effects** — vary labelled training set size at fixed *N* = 10 000.

## eval+training harness in `evaluation/`
the evaluation harness, for both training and evaluation. README coming soon for how to add methods
