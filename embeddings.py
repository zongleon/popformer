import numpy as np
from tqdm import tqdm
import torch
from models import HapbertaForMaskedLM
from datasets import load_from_disk, DatasetDict
from collators import HaploSimpleDataCollator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

dsd = {}
pops = ["CEU", "CHB", "YRI", "ESN", "CHS", "GBR"]
for pop in pops:
    ds = load_from_disk(f"SEL/tokenized_{pop}")
    dsd[pop] = ds.take(64)
dsd = DatasetDict(dsd)

collator = HaploSimpleDataCollator(subsample=None)

# plot first 2 PCs
colors = {'CEU': 'tab:blue', 'CHB': 'tab:orange', 'YRI': 'tab:green', 
          'ESN': 'tab:purple', 'CHS': 'tab:red', 'GBR': 'tab:brown'}
# pop_colors = [colors[label] for label in pops]
# pop_colors = np.array(pop_colors).repeat(32, axis=0)

# model = HapbertaForMaskedLM.from_pretrained(
#     "./models/hapberta2d/",
#     torch_dtype=torch.bfloat16
# )
model = HapbertaForMaskedLM.from_pretrained(
    "./models/pt4/",
    torch_dtype=torch.float16
)
model.to("cuda")
model.eval()
# model.compile()

def preds():
    batch_size = 4
    lbls = []
    all_embeds = []
    with torch.inference_mode():
        for pop in pops:
            embeds = []
            ds = dsd[pop]
            for i in tqdm(range(0, len(ds), batch_size)):
                batch_samples = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
                batch = collator(batch_samples)
                # Move tensors to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to("cuda")
                
                output = model(batch["input_ids"], batch["distances"], batch["attention_mask"],
                            labels=None,
                            return_hidden_states=True)
                
                hidden = output["hidden_states"].mean(axis=2)  # (batch_size, n_haps, n_snps, hidden_size)
                n_haps = hidden.shape[1]
                embeds.append(hidden.permute(1, 0, 2).reshape(n_haps, -1).cpu().numpy())  # (n_haps, batch_size * hidden_size)
                # hidden = output["hidden_states"].mean(axis=(0, 2)).cpu().numpy()
                # n_haps = hidden.shape[0]
                # embeds.append(hidden)

            embeds = np.concatenate(embeds, axis=1)
            all_embeds.append(embeds)
            lbls.extend([pop] * n_haps)
    
    return np.concatenate(all_embeds, axis=0), lbls
    
# pca on embeds
def pca(embeds, pop_colors):
    embed_pca = PCA(n_components=10)
    embed_pca.fit(embeds)

    reduced_embeds = embed_pca.transform(embeds)

    # plot explained variance
    # var = embed_pca.explained_variance_ratio_
    # plt.figure(figsize=(10, 5))
    # plt.bar(range(len(var)), var)
    # plt.xlabel("Principal Component")
    # plt.ylabel("Variance Explained")
    # plt.title("PCA - Explained Variance")
    # plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], c=pop_colors)
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=p, markerfacecolor=color, markersize=10)
            for p, color in colors.items()]
    plt.legend(handles=handles, title="Population")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("pca")
    plt.savefig("figs/embeds_pca.png", dpi=300)

def um(embeds, pop_colors):
    # umap on embeds
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(embeds)

    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=pop_colors)
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=pop, markerfacecolor=color, markersize=10)
            for pop, color in colors.items()]
    plt.legend(handles=handles, title="Population")
    plt.xlabel("umap 1")
    plt.ylabel("umap 2")
    plt.title("umap")
    plt.savefig("figs/embeds_umap.png", dpi=300)

def attns(model):
    inputs = collator([ds[0]])

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to("cuda")

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["distances"], inputs["attention_mask"], return_attentions=True)

    row_attn, col_attn = outputs["attentions"]

    idx = -1
    row_attn, col_attn = row_attn[idx].to(torch.float16).cpu().numpy(), col_attn[idx].to(torch.float16).cpu().numpy()

    haps = inputs["input_ids"].cpu().numpy()[0]
    haps[(haps > 1)] = 0
    plt.figure(figsize=(8, 6))
    plt.imshow(haps[:, :], aspect='auto', cmap='Greys', interpolation="none")
    plt.xlabel("SNP Position")
    plt.ylabel("Haplotype")
    plt.title("Original Haplotype Matrix")
    plt.colorbar(label="Allele")
    plt.savefig("figs/ex_matrix.png", dpi=300)

    # Plot row attention maps for each head in layer idx
    num_heads = row_attn.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(row_attn[i, 0, :256, :256], aspect='auto', cmap='viridis')
        ax.set_title(f'Head {i}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("figs/snp_attn.png", dpi=300)

    # Plot col attention maps for each head in layer idx
    snp_pos = 4
    num_heads = col_attn.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(col_attn[i, snp_pos, 0], aspect='auto', cmap='viridis')
        ax.set_title(f'Head {i}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("figs/hap_attn_snp_4.png", dpi=300)


embeds, lbls = preds()
pop_colors = [colors[label] for label in lbls]
pca(embeds, pop_colors)
um(embeds, pop_colors)