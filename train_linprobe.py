from matplotlib.axes import Axes
import numpy as np
from transformers import RobertaConfig
from models import HapbertaForMaskedLM
from datasets import load_from_disk
from collators import HaploSimpleDataCollator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

OUTPUT = "./models/hapberta2d_mae"
DATASET_PATH = "dataset2/tokenized"
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-3

dataset = load_from_disk(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

config = RobertaConfig.from_pretrained(OUTPUT)
config.encoder_only = True
model = HapbertaForMaskedLM.from_pretrained(OUTPUT, config=config, torch_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
for param in model.roberta.parameters():
    param.requires_grad = False

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.probe = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, haps, snps, hidden)
        return self.probe(x)

probe = LinearProbe(config.hidden_size, 2).to(device, torch.bfloat16)

# 4. Data collator and loader
data_collator = HaploSimpleDataCollator(subsample=64, mlm_probability=0.0, whole_snp_mask_probability=0.4)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

# 5. Optimizer
optimizer = torch.optim.SGD(probe.parameters(), lr=LR)
loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 4.0]).to(device, dtype=torch.bfloat16), ignore_index=-100)

# 6. Training loop (encoder only)
for epoch in range(EPOCHS):
    probe.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        distances = batch["distances"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            # hidden, _ = model.roberta(input_ids, distances)
            outputs = model(input_ids, distances, attention_mask,
                            return_hidden_states=True)
            hidden = outputs["hidden_states"]

        logits = probe(hidden)  # (batch, haps, snps, vocab)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

        if pbar.n % 50 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (preds[mask] == labels[mask]).sum().item()
                total = mask.sum().item()
                acc = correct / total if total > 0 else 0.0
                print(f"Step {pbar.n}: Accuracy = {acc:.4f}")

        if pbar.n != 0 and pbar.n % 100 == 0:
            break

# After training loop, add evaluation
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

probe.eval()
total_loss = 0.0
total_correct = 0
total_count = 0

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        distances = batch["distances"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # hidden, _ = model.roberta(input_ids, distances)
        outputs = model(input_ids, distances, attention_mask,
                        return_hidden_states=True)
        hidden = outputs["hidden_states"]
        logits = probe(hidden)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item() * input_ids.size(0)

        preds = logits.argmax(dim=-1)
        print(np.unique(preds.cpu().numpy()))
        mask = labels != -100
        np.savetxt("mask.txt", mask[0].cpu().numpy(), fmt="%d")
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        total_correct += correct
        total_count += total

        # input_ids: (batch, haps, snps)
        ax0: Axes
        ax1: Axes
        ax2: Axes
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 10))
        
        img = input_ids[0].cpu().numpy()
        def color(img):
            img = img[:50]
            # Create a color image: 0->white, 1->black, 4->red
            color_img = np.stack([img, img, img], axis=-1).astype(float)
            # Set all to white
            color_img[:] = 1.0
            # Set 0 to white, 1 to black, 4 to red
            color_img[img == 0] = [1, 1, 1]
            color_img[img == 1] = [0, 0, 0]
            color_img[img == 2] = [0, 1, 0]
            color_img[img == 3] = [0, 0, 1]
            color_img[img == 4] = [1, 0, 0]
            color_img[img == 5] = [0, 0, 0]
            return color_img

        ax0.imshow(color(img), aspect='auto', interpolation="none")
        ax0.set_title("masked")
        ax0.set_ylabel("Haplotypes")

        pred_img = preds[0].cpu().numpy()
        ax1.imshow(color(pred_img), aspect='auto', cmap='Greys', interpolation="none")
        ax1.set_title("predicted")

        # Show ground truth: input_ids with masked id 4 replaced by labels
        gt_img = input_ids[0].clone()
        mask = (gt_img == 4)
        gt_img[mask] = labels[0][mask]
        gt_img = gt_img.cpu().numpy()
        ax2.imshow(color(gt_img), aspect='auto', cmap='Greys', interpolation="none")
        ax2.set_title("ground truth")

        plt.savefig("figs/ex_maskedrecreation.png", dpi=300, bbox_inches="tight")

        break

eval_loss = total_loss / len(eval_dataset)
eval_acc = total_correct / total_count if total_count > 0 else 0.0
print(f"Eval loss: {eval_loss:.4f}, Eval accuracy: {eval_acc:.4f}")

