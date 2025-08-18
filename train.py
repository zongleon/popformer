import numpy as np
from model import *
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def create_random_mask(data_shape, mask_prob=0.3, device='cpu'):
    """
    Random SNP mask across haplotypes.
    Args:
        data_shape: (n, n_hap, n_snp)
        mask_prob: fraction of SNPs to mask
    Returns:
        mask: boolean tensor of shape (n, n_hap, n_snp)
    """
    mask = torch.rand(data_shape, device=device) < mask_prob
    return mask


def compute_class_weights(targets):
    """
    Compute class weights for imbalanced data.
    Args:
        targets: (n, n_hap, n_snp) tensor of 0/1
    Returns:
        weights: tensor of shape (2,)
    """
    flat = targets.view(-1)
    n0 = (flat == 0).sum().item()
    n1 = (flat == 1).sum().item()
    total = n0 + n1
    # Avoid division by zero
    w0 = total / (2 * n0) if n0 > 0 else 1.0
    w1 = total / (2 * n1) if n1 > 0 else 1.0
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    return weights


def train_epoch(model, head, optimizer, data, targets, batch_size=16, device='cuda'):
    """
    Single training epoch for summary statistic prediction (regression).
    """
    model.train()
    head.train()
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    n_batches = 0

    for xb, y in loader:
        xb, y = xb.to(device), y.to(device)
        optimizer.zero_grad()
        reps = model(xb)  # (b, n_hap, n_snp, hidden)
        # Pool over SNPs and haplotypes to get a single representation per sample
        pooled = reps.mean(dim=(1, 2))  # (b, hidden)
        preds = head(pooled)  # (b, n_stats)
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

        if n_batches % 10 == 0:
            print(f"Batch {n_batches}: loss={loss.item():.4f}")

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(model, head, data, targets, batch_size=16, device='cuda'):
    """
    Evaluate summary statistic prediction (regression MSE).
    """
    model.eval()
    head.eval()
    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, y in loader:
            xb, y = xb.to(device), y.to(device)
            reps = model(xb)
            pooled = reps.mean(dim=(1, 2))  # (b, hidden)
            preds = head(pooled)  # (b, n_stats)
            loss = F.mse_loss(preds, y)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss

def apply_input_mask(data, mask, mask_value=0):
    """
    Mask the input allele channel at masked positions.
    Args:
        data: (n, n_hap, n_snp, 2) tensor
        mask: (n, n_hap, n_snp) boolean tensor, True for masked positions
        mask_value: value to use for masked positions (should not be a real allele)
    Returns:
        masked_data: copy of data with masked positions set to mask_value in allele channel
    """
    data_masked = data.clone()
    data_masked[..., 0][mask] = mask_value
    return data_masked

# Example usage for summary statistic prediction
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = 2
hidden_dim = 128
n_heads = 4
n_layers = 4
n_stats = 2  # e.g., allele frequency and heterozygosity

model = SiteHapModel(input_dim, hidden_dim, n_heads, n_layers).to(device)
# The head should output n_stats values (regression)
head = torch.nn.Linear(hidden_dim, n_stats).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-5)

def get_data(path: str):
    data = np.load(path)[:10000]
    data[:, :, :, 0][data[:, :, :, 0] == -1] = 0  # replace -1 with 0
    return torch.tensor(data, dtype=torch.float32, device=device)

def compute_summary_stats(data):
    # data: (n, n_hap, n_snp, 2)
    # Compute allele frequency and heterozygosity for each sample
    allele = data[..., 0]  # (n, n_hap, n_snp)
    # allele frequency: mean over all haplotypes and SNPs
    freq = allele.float().mean(dim=(1, 2))  # (n,)
    # heterozygosity: mean of 2*p*(1-p) over all SNPs and haps
    p = allele.float().mean(dim=2)  # (n, n_hap)
    het = (2 * p * (1 - p)).mean(dim=1)  # (n,)
    stats = torch.stack([freq, het], dim=1)  # (n, 2)
    return stats

train_data = get_data("../disc-interpret/dataset-CEU/X.npy")
train_targets = compute_summary_stats(train_data)
test_data = get_data("../disc-interpret/dataset-CHB/X.npy")
test_targets = compute_summary_stats(test_data)

for epoch in range(3):
    loss = train_epoch(model, head, optimizer, train_data, train_targets, batch_size=32, device=device)
    print(f"Epoch {epoch+1}: train loss={loss:.4f}")

    test_loss = evaluate(model, head, test_data, test_targets, batch_size=32, device=device)
    print(f"Test summary stat MSE: {test_loss:.4f}")

# save pre-trained model
torch.save(model.state_dict(), "sitehap_pretrain.pth")
