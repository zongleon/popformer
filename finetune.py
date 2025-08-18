import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import *
import torch.nn.functional as F

# data: (n, n_hap, n_snp, 2)
def get_data(x: str, y: str, n=10000):
    data = np.load(x)
    y = np.load(y)
    shuf = np.random.randint(0, data.shape[0], n)
    data, y = data[shuf], y[shuf]
    data[:, :, :, 0][data[:, :, :, 0] == -1] = 0  # replace -1 with 0
    return torch.tensor(data, dtype=torch.float32, device="cuda"), torch.tensor(y, dtype=torch.float32, device="cuda")

train_data, train_labels = get_data("../disc-interpret/dataset-CEU/X.npy", "../disc-interpret/dataset-CEU/y.npy")
test_data, test_labels = get_data("../disc-interpret/dataset-CHB/X.npy", "../disc-interpret/dataset-CHB/y.npy")

dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SiteHapModel(input_dim=2, hidden_dim=64, n_heads=4).to("cuda")
model.load_state_dict(torch.load("sitehap_pretrain.pth"))

# Freeze model parameters
# for param in model.parameters():
#     param.requires_grad = False

# backbone: pretrained SiteHapModel
# head: SampleLevelHead
sample_head = SampleLevelHead(hidden_dim=64, output_dim=1).to("cuda")
optimizer = torch.optim.Adam(list(model.parameters()) + list(sample_head.parameters()), lr=1e-4)

print(f"# parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in sample_head.parameters() if p.requires_grad)}")

# Use BCEWithLogitsLoss for binary classification
criterion = torch.nn.BCEWithLogitsLoss()

model.train()
sample_head.train()
i = 0
for xb, y in train_loader:  # xb: (b, n_hap, n_snp, 2), y: (b,)
    optimizer.zero_grad()
    reps = model(xb)  # (b, n_hap, n_snp, hidden)
    preds = sample_head(reps)  # (b, 1)
    loss = criterion(preds.squeeze(), y)
    loss.backward()
    optimizer.step()
    i += 1

    if (i + 1) % 10 == 0:
        print(f"Step {i+1}, Loss: {loss.item():.4f}")


# For binary classification, use sigmoid and threshold at 0.5
pred_labels = (torch.sigmoid(preds.squeeze()) > 0.5).float()
acc = (pred_labels == y).float().mean()

# eval
model.eval()
sample_head.eval()
with torch.no_grad():
    for xb, y in test_loader:  # xb: (b, n_hap, n_snp, 2), y: (b,)
        reps = model(xb)  # (b, n_hap, n_snp, hidden)
        preds = sample_head(reps)  # (b, 1)
        test_loss = criterion(preds.squeeze(), y)
        pred_labels = (torch.sigmoid(preds.squeeze()) > 0.5).float()
        test_acc = (pred_labels == y).float().mean()
        # print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc.item():.4f}")
print(f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc.item():.4f}")


# save
torch.save(model.state_dict(), "sitehap_finetune.pth")
torch.save(sample_head.state_dict(), "sitehap_head_finetune.pth")
