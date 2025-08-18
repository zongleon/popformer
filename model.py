import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.embedding(x)

class SiteBlock(nn.Module):
    """
    Site-dimension module. For modeling along the sequence dimension.
    """
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(hidden_dim, n_heads, batch_first=True)

    def forward(self, x):
        # x like (b, n_hap, n_snp, hidden)
        # separate haps
        b, n, s, h = x.shape
        x = x.view(b * n, s, h)
        x = self.attn(x)
        return x.view(b, n, s, h)
    

class HapBlock(nn.Module):
    """
    Haplotype-dimension module with permutation-equivariant pooling.
    Returns haplotype-resolved features, enriched with pooled population context.
    """
    def __init__(self, dim, seed_dim=1, n_heads=4):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, seed_dim, dim))
        nn.init.xavier_uniform_(self.S)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.proj_back = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (b, n_hap, n_snp, hidden)
        b, n, s, h = x.shape
        x_reshaped = x.view(b * s, n, h)  # treat each site separately

        # pooled population summary
        S_rep = self.S.repeat(b * s, 1, 1)  # (b*s, seed_dim, hidden)
        pooled, _ = self.attn(S_rep, x_reshaped, x_reshaped)  # (b*s, seed_dim, hidden)
        pooled = pooled.mean(dim=1)  # aggregate seed tokens -> (b*s, hidden)

        # broadcast back to haplotypes as a residual
        proj = self.norm(self.proj_back(pooled))  # (b*s, hidden)
        x_out = x_reshaped + proj.unsqueeze(1)  # (b*s, n, hidden)
        return x_out.view(b, n, s, h)


class SiteHapModel(nn.Module):
    """Main model, aggregating first across SNPs and then across haps."""
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers=1):
        super().__init__()
        self.embedding = Embedding(input_dim, hidden_dim)

        # repeat layers
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(SiteBlock(hidden_dim, n_heads))
            self.layers.append(HapBlock(hidden_dim))

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SNPModelHead(nn.Module):
    """Haplotype-resolved masked SNP prediction decoder"""
    def __init__(self, hidden_dim, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (b, n_hap, n_snp, hidden)
        logits = self.fc(x)  # (b, n_hap, n_snp, output_dim)
        return logits


class SampleLevelHead(nn.Module):
    """Head for predicting sample-level features, like a simulation parameter.
    Predict a logit per sample in batch, like (b, output)."""
    def __init__(self, hidden_dim, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (b, n_hap, n_snp, hidden)
        logits = self.fc(x)  # (b, n_hap, n_snp, output_dim)
        return logits
    

class SampleLevelHead(nn.Module):
    """
    Predicts a sample-level feature from haplotype embeddings.
    Aggregates haplotypes and sites via mean (or sum) pooling.
    """
    def __init__(self, hidden_dim, output_dim=1, pool='mean'):
        super().__init__()
        self.pool = pool
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (batch, n_hap, n_snp, hidden)
        returns: (batch, output_dim)
        """
        if self.pool == 'mean':
            # mean over haplotypes and SNPs
            x_pooled = x.mean(dim=(1, 2))
        elif self.pool == 'sum':
            x_pooled = x.sum(dim=(1, 2))
        else:
            raise ValueError(f"Unknown pooling method {self.pool}")
        return self.fc(x_pooled)
