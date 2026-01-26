"""
Generate, store, and visualize embeddings from pre-trained popformer models.
Do the populations separate?
"""

import os
import sys
from cyvcf2 import VCF
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig
from popformer.models import PopformerForMaskedLM, PopformerForWindowClassification
from datasets import load_from_disk
from popformer.collators import HaploSimpleDataCollator
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import theme


def preds(model, ds):
    batch_size = 64
    all_embeds = np.zeros((len(vcf.samples), len(ds)), dtype=np.float16)
    # all_embeds = np.zeros(
    #     (len(vcf.samples), len(ds), model.config.hidden_size), dtype=np.float16
    # )
    print("Total samples:", len(ds))
    print("  with batch size 64:", (len(ds) + batch_size - 1) // batch_size)

    dataloader = DataLoader(
        ds, batch_size=batch_size, collate_fn=collator, num_workers=4
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            i = batch_idx * batch_size
            ni = min(i + batch_size, len(ds))

            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to("cuda")

            for hap in tqdm(range(0, len(vcf.samples) * 2, 2)):
                output = model(
                    batch["input_ids"][:, hap : hap + 2, :],
                    batch["distances"],
                    batch["attention_mask"],
                    labels=None,
                    return_hidden_states=True,
                )

                # mean over haps, snps, hidden dim
                hidden = output["hidden_states"].mean(axis=(1, 2, 3))
                all_embeds[hap // 2, i:ni] = hidden.cpu().numpy()

                # mean over haps, snps
                # hidden = output["hidden_states"].mean(axis=(1, 2))
                # all_embeds[hap // 2, i:ni, :] = hidden.cpu().numpy()

    # all_embeds = all_embeds.mean(axis=1)  # mean over windows

    return all_embeds


def preds_window(model, ds):
    batch_size = 8
    all_embeds = np.zeros((len(ds), model.config.hidden_size), dtype=np.float16)
    print("Total samples:", len(ds))
    print(f"  with batch size {batch_size}:", (len(ds) + batch_size - 1) // batch_size)

    dataloader = DataLoader(
        ds, batch_size=batch_size, collate_fn=collator, num_workers=4
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            i = batch_idx * batch_size
            ni = min(i + batch_size, len(ds))

            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to("cuda")

            output = model(
                batch["input_ids"],
                batch["distances"],
                batch["attention_mask"],
                labels=None,
                return_hidden_states=True,
            )

            # mean over haps, snps
            hidden = output["hidden_states"].mean(axis=(1, 2))
            all_embeds[i:ni, :] = hidden.cpu().numpy()

    return all_embeds


# pca on embeds
def pca(embeds, lbls, figpath, legend_title="POP", continuous=False):
    embed_pca = PCA(random_state=SEED, n_components=16)
    embed_pca.fit(embeds)

    reduced_embeds = embed_pca.transform(embeds)

    embeds_df = pd.DataFrame(reduced_embeds, columns=[f"PC{i + 1}" for i in range(16)])
    embeds_df[legend_title] = lbls

    # plot explained variance
    var = embed_pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(var)), var)
    plt.xlabel("pc #")
    plt.ylabel("variance")
    plt.savefig(f"figs/embeds/pca_ev_{legend_title}.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 10), layout="constrained")
    s = sns.scatterplot(
        data=embeds_df,
        x="PC1",
        y="PC2",
        hue=legend_title,
        palette="viridis" if continuous else theme.pop_to_color,
        # legend="full",
    )
    s.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    sns.despine()

    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("pca")
    # plt.legend(title=legend_title, bbox_to_anchor=(1.03, 1))

    plt.savefig(figpath, dpi=300)


def attns(model):
    inputs = collator([ds[0]])

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to("cuda")

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            inputs["distances"],
            inputs["attention_mask"],
            return_attentions=True,
        )

    row_attn, col_attn = outputs["attentions"]

    idx = -1
    row_attn, col_attn = (
        row_attn[idx].to(torch.float16).cpu().numpy(),
        col_attn[idx].to(torch.float16).cpu().numpy(),
    )

    haps = inputs["input_ids"].cpu().numpy()[0]
    haps[(haps > 1)] = 0
    plt.figure(figsize=(8, 6))
    plt.imshow(haps[:, :], aspect="auto", cmap="Greys", interpolation="none")
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
        im = ax.imshow(row_attn[i, 0, :256, :256], aspect="auto", cmap="viridis")
        ax.set_title(f"Head {i}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("figs/snp_attn.png", dpi=300)


if __name__ == "__main__":
    SEED = 42
    mode = sys.argv[1]

    if mode == "continent" or mode == "pop":
        vcf = VCF(
            "/bigdata/smathieson/1000g-share/VCF/ALL.chr1.snps.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
        )
        df = pd.read_csv("data/igsr_samples.tsv", sep="\t")

        pops = [
            df[df["Sample name"] == s]["Population code"].values[0] for s in vcf.samples
        ]

        pop_to_continent = {
            pop: df[df["Population code"] == pop]["Superpopulation code"].values[0]
            for pop in list(set(pops))
        }

        # double every pop (haplotype)
        pops = [[p, p] for p in pops]
        pops = np.array(pops).flatten()

        # Apply continent mapping if option is enabled
        continents = [pop_to_continent.get(p, p) for p in pops]

        ds = load_from_disk("data/dataset/genome_ALL.chr1")

        collator = HaploSimpleDataCollator(subsample=None)

        model = PopformerForMaskedLM.from_pretrained(
            "./models/popf-small", torch_dtype=torch.float16
        )
        model.to("cuda")
        model.eval()

        if os.path.exists("embeds.npy"):
            embeds = np.load("embeds.npy")
        else:
            embeds = preds(model, ds)
            np.save("embeds.npy", embeds)

        if mode == "continent":
            # map labels to continent
            lbls = continents[::2]
        else:
            lbls = pops[::2]

        lbls = np.array(lbls)

        mask = lbls != "EUR,AFR"
        embeds = embeds[mask]
        lbls = lbls[mask]

        pca(embeds, lbls, "figs/embeds/pca.png")
    elif mode == "selection":
        ds = (
            load_from_disk("data/dataset/pan_4_test").shuffle(42)  # .take(2048)
        )
        embeds_path = "embeds_sel_init.npy"

        collator = HaploSimpleDataCollator(subsample=(64, 64), subsample_type="diverse")

        # model = PopformerForWindowClassification.from_pretrained(
        #     "./models/selbin-ft", torch_dtype=torch.float16
        # )

        model = PopformerForMaskedLM(AutoConfig.from_pretrained("./models/popf-small"))

        # model = PopformerForMaskedLM.from_pretrained(
        #     "./models/popf-small", torch_dtype=torch.float16
        # )
        model.to("cuda")
        model.eval()

        if os.path.exists(embeds_path):
            embeds = np.load(embeds_path)
        else:
            embeds = preds_window(model, ds)
            np.save(embeds_path, embeds)
        lbls = np.array(ds["s"])
        # lbls = np.array(ds["low_mut"])
        # lbls = (np.array(ds["label"]) == 0) & (np.array(ds["s"]) > 0)
        # lbls = lbls.astype(int)

        # pca(embeds, lbls, "figs/embeds/pca.png", legend_title="s", continuous=True)
        pca(
            embeds,
            lbls,
            "figs/embeds/pca.png",
            legend_title="s",
            continuous=True,
        )
    else:
        raise ValueError("Unknown mode: use 'continent' or 'pop' or 'selection'")
