"""
Re-implement of Faster-NN model training for evaluation.
"""

from datasets import load_from_disk
from torch.utils.data import DataLoader

import os
import sys

sys.path.append("analysis/")
from evaluation.models.fasternn import FasterNNModel

test_size = 0.05
if len(sys.argv) > 2:
    test_size = float(sys.argv[2])

model = FasterNNModel(
    model_path=f"models/fasternn/fasternn_{os.path.basename(os.path.normpath(sys.argv[1]))}-{test_size}.pt",
    model_name="FASTER-NN",
    from_init=True,
)

data = load_from_disk(sys.argv[1])
split_data = data.train_test_split(
    test_size=test_size, seed=42
)  # , stratify_by_column="label")
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

model.train(train_loader, val_loader, epochs=100, lr=1e-4, patience=3)
