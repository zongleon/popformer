# Popformer

An axial attention transformer for haplotype matrices. Self-supervised pre-training
on masked haplotype reconstruction, with downstream evaluation on natural selection
detection, genotype imputation, and population classification.

**preprint out now:** [Popformer: Learning general signatures of positive selection with a self-supervised transformer](https://www.biorxiv.org/content/10.64898/2026.03.06.710163v1)

## Model Weights

Available via Huggingface:

- Pre-trained model: [leonzong/popf-small](https://huggingface.co/leonzong/popf-small)
- Fine-tuned selection model: [leonzong/popf-ft-selection-CEU](https://huggingface.co/leonzong/popf-ft-selection-CEU)

## Setup

### Installation

```bash
# use a venv
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Inference: genome-wide selection scan on a VCF

### 1. preprocess

Convert a VCF (or HDF5 / tree sequence) into windowed, tokenized input:

```bash
# From a VCF or HDF5 file
python -m popformer.dataset input.vcf.gz tokenized_dataset/ --num_snps 512 --max_haps 256 --window_size 50000 --window_jump 50000

# With a callable-regions BED mask, restricted to one chromosome
python -m popformer.dataset input.vcf.gz tokenized_dataset/ --bed_file callable_regions.bed --frac_callable 0.5 --chrom 2

# From simulated tree sequences (directory of .trees files)
python -m popformer.dataset simulations/ tokenized_dataset/
```

### 2. Run inference

Use `sweep.py` to run inference and save/plot predictions from the command line:

```bash
# Save per-window logits to an .npz file
python sweep.py --data tokenized_dataset/ --model leonzong/popf-ft-selection-CEU --save_logits preds/scan.npz --subsample 64

# Generate a Manhattan plot from saved predictions
python sweep.py --logits_path preds/scan.npz --plot_preds figs/manhattan.png --smooth_window 7

# Save extracted embeddings
python sweep.py --data tokenized_dataset/ --model lzong/popf-ft-selection-CEU --save_features preds/features.npz
```

Or, manually in Python:

```python
import torch
import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from popformer.models import PopformerForWindowClassification
from popformer.collators import HaploSimpleDataCollator

# Load model and data
model = PopformerForWindowClassification.from_pretrained(
    "leonzong/popf-ft-selection-CEU", torch_dtype=torch.float16
)
model.eval().cuda()

data = load_from_disk("tokenized_dataset/")
collator = HaploSimpleDataCollator(subsample=(64, 64), subsample_type="diverse")
loader = DataLoader(data, batch_size=4, num_workers=4, collate_fn=collator)

all_preds = []
with torch.inference_mode():
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = model(batch["input_ids"], batch["distances"], batch["attention_mask"])
        probs = torch.softmax(output["logits"], dim=-1)[:, 1]  # p(selection)
        all_preds.append(probs.cpu().numpy())

preds = np.concatenate(all_preds)
```

## Training

end-to-end training example coming soon

## Project structure

```
popformer/          popformer model + collator + dataset tools
analysis/
  train/            training scripts
  evaluation/       evaluation harness
  test_*.py         analysis scripts + figures
  scripts/          various data preprocessing
sweep.py
models/
data/
```
