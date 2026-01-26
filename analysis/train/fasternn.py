"""
Re-implement of Faster-NN model training for evaluation.
"""

from datasets import load_from_disk
from torch.utils.data import DataLoader

import os
import sys

sys.path.append("analysis/")
from evaluation.models.fasternn import FasterNNModel

model = FasterNNModel(
    model_path=f"models/fasternn/fasternn_{os.path.basename(os.path.normpath(sys.argv[1]))}.pt",
    model_name="FASTER-NN",
    from_init=True,
)

data = load_from_disk(sys.argv[1]).shuffle(42)
split_data = data.train_test_split(test_size=0.05)  # , stratify_by_column="label")
train_data = split_data["train"]
val_data = split_data["test"]


train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=model.preprocess,
)
val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=model.preprocess,
)

model.train(train_loader, val_loader, epochs=100, lr=1e-4, patience=5)
