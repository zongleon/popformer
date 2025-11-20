"""
Re-implement of Faster-NN model training for evaluation.
"""
from datasets import load_from_disk
from torch.utils.data import DataLoader

import sys
sys.path.append("analysis/")
from evaluation.models.schrider_resnet import SchriderResnet

model = SchriderResnet(model_path="models/schrider_resnet/schrider_resnet.pt", model_name="resnet34", from_init=True)

data = load_from_disk("data/dataset/pan_4_train_balanced")
split_data = data.train_test_split(test_size=0.1)
train_data = split_data["train"]
val_data = split_data["test"]

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=model.preprocess,
)
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=model.preprocess,
)

model.train(train_loader, val_loader, epochs=50, lr=1e-4, patience=10)