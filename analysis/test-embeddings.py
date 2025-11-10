"""
Generate, store, and visualize embeddings from pre-trained popformer models.
Do the populations separate?
"""

from cyvcf2 import VCF
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from popformer.models import PopformerForMaskedLM
from datasets import load_from_disk
from popformer.collators import HaploSimpleDataCollator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

SEED = 42
use_continent = False
ext = "sup" if use_continent else "pop"
cmap = "tab10" if use_continent else "tab20"

vcf = VCF("/bigdata/smathieson/1000g-share/VCF/ALL.chr1.snps.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz")
df = pd.read_csv("dataset4/1000g_samples.tsv", sep="\t")

pops = [df[df["Sample name"] == s]["Population code"].values[0] for s in vcf.samples]

pop_to_continent = {pop: df[df["Population code"] == pop]["Superpopulation code"].values[0] for pop in df["Population code"].unique()}

# Apply continent mapping if option is enabled
if use_continent:
    pops = [pop_to_continent.get(p, p) for p in pops]

# double every pop (haplotype)
pops = [[p, p] for p in pops]
pops = np.array(pops).flatten()

ds = load_from_disk("SEL/tokenized_ALL.chr1.snps")

collator = HaploSimpleDataCollator(subsample=None)

model = PopformerForMaskedLM.from_pretrained(
    "./models/pt2/",
    torch_dtype=torch.float16
)
model.to("cuda")
model.eval()
# model.compile()

def preds():
    batch_size = 64
    all_embeds = np.zeros((len(vcf.samples), len(ds)), dtype=np.float16)
    # all_embeds = np.zeros((len(vcf.samples), len(ds), model.config.hidden_size), dtype=np.float16)
    # all_embeds = np.zeros((len(vcf.samples), len(ds), 34))

    with torch.inference_mode():
        for i in range(0, len(ds), batch_size):
            ni = min(i + batch_size, len(ds))
            batch_samples = [ds[j] for j in range(i, ni)]
            batch = collator(batch_samples)

            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to("cuda")

            for hap in tqdm(range(0, len(vcf.samples) * 2, 2)):
            
                output = model(batch["input_ids"][:, hap:hap+2, :], 
                               batch["distances"], 
                               batch["attention_mask"], 
                               labels=None, 
                               return_hidden_states=True)

                # mean over haps, snps, hidden dim
                hidden = output["hidden_states"].mean(axis=(1, 2, 3))
                all_embeds[hap // 2, i:ni] = hidden.cpu().numpy()

                # mean over haps, snps
                # hidden = output["hidden_states"].mean(axis=(1, 2))
                # all_embeds[hap // 2, i:ni, :] = hidden.cpu().numpy()

                # mean over haps, hidden dim
                # hidden = output["hidden_states"].mean(axis=(1, 3))
                # all_embeds[hap // 2, i:ni, :] = hidden.cpu().numpy()

                # mean over haps, hidden dim
                # hidden = output["hidden_states"].mean(axis=(1, 3))
                # hidden = hidden.flatten()
                # start = i * 258
                # end = ni * 258
                # all_embeds[hap // 2, start:end] = hidden.detach().cpu().numpy().ravel()

    # all_embeds = all_embeds.mean(axis=1)
    # all_embeds = all_embeds.reshape(all_embeds.shape[0], -1)
    # all_embeds = all_embeds.reshape(-1, all_embeds.shape[-1])
    
    np.save("embeds.npy", all_embeds)
    print(all_embeds.shape)
    return all_embeds
    
# pca on embeds
def pca(embeds, lbls, figpath):
    embed_pca = PCA(random_state=SEED, n_components=16)
    embed_pca.fit(embeds)

    reduced_embeds = embed_pca.transform(embeds)

    # plot explained variance
    var = embed_pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(var)), var)
    plt.xlabel("pc #")
    plt.ylabel("variance")
    plt.savefig("figs/pca_ev.png", dpi=300, bbox_inches="tight")

    unique_pops = sorted(set(lbls), key=lambda p: pop_to_continent.get(p, p))
    unique_pops.remove("IBS,MSL")

    markers = ["o", "s", "^", "+", "x", "v"]

    colors = {}
    continents = sorted(set([pop_to_continent.get(p, p) for p in unique_pops]))
    for continent in continents:
        pops_in_cont = [p for p in unique_pops if pop_to_continent[p] == continent]
        cmap_cont = plt.get_cmap('tab10')
        for i, pop in enumerate(pops_in_cont):
            colors[pop] = cmap_cont(i)

    plt.figure(figsize=(10, 10), layout="constrained")
    for i, pop in enumerate(unique_pops):
        pop_embeds = reduced_embeds[np.array(lbls) == pop]

        marker = "o"
        color = None
        if not use_continent:
            marker = markers[continents.index(pop_to_continent[pop])]
            color = colors[pop]

        plt.scatter(pop_embeds[:, 0], pop_embeds[:, 1], marker=marker, color=color, label=pop)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("pca")
    plt.legend(title="POP", bbox_to_anchor=(1.03, 1))

    plt.savefig(figpath, dpi=300)

def um(embeds, lbls, figpath):
    # umap on embeds
    # Reduce dimensionality with PCA first
    pca_reducer = PCA(random_state=SEED, n_components=16)
    embeds_pca = pca_reducer.fit_transform(embeds)

    reducer = umap.UMAP(random_state=SEED, min_dist=0.5)
    embedding = reducer.fit_transform(embeds_pca)

    unique_pops = sorted(set(lbls), key=lambda p: pop_to_continent.get(p, p))
    unique_pops.remove("IBS,MSL")
    markers = ["o", "s", "^", "+", "x", "v"]

    colors = {}
    continents = sorted(set([pop_to_continent.get(p, p) for p in unique_pops]))
    for continent in continents:
        pops_in_cont = [p for p in unique_pops if pop_to_continent[p] == continent]
        cmap_cont = plt.get_cmap('tab10')
        for i, pop in enumerate(pops_in_cont):
            colors[pop] = cmap_cont(i)

    plt.figure(figsize=(10, 10), layout="constrained")
    for i, pop in enumerate(unique_pops):
        pop_embeds = embedding[np.array(lbls) == pop]
        
        marker = "o"
        color = None
        if not use_continent:
            marker = markers[continents.index(pop_to_continent[pop])]
            color = colors[pop]

        plt.scatter(pop_embeds[:, 0], pop_embeds[:, 1], label=pop, color=color, marker=marker)

    plt.xlabel("umap 1")
    plt.ylabel("umap 2")
    plt.title("umap")

    plt.legend(title="POP", bbox_to_anchor=(1.03, 1), loc='upper left')

    plt.savefig(figpath, dpi=300)

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


# embeds = preds()

embeds = np.load("embeds.npy")

if use_continent:
    # map labels to continent
    lbls = [pop_to_continent.get(p, p) for p in pops[::2]]
    # lbls = np.repeat(lbls, len(ds))
else:
    lbls = pops[::2]


lbls = np.array(lbls)

pca(embeds, lbls, f"figs/embed_pca2_{ext}_128.png")
um(embeds, lbls, f"figs/embed_umap2_{ext}_128.png")