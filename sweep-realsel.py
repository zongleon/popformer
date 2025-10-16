import contextlib
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import HapbertaForSequenceClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def sweep(dataset, model, save_preds_path=None):
    data = load_from_disk(dataset)

    model = HapbertaForSequenceClassification.from_pretrained(
        model,
        torch_dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=(32, 32))

    loader = DataLoader(
        data,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    preds = []

    with torch.inference_mode():
        for batch in tqdm(loader):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            output = model(batch["input_ids"], 
                            batch["distances"], 
                            batch["attention_mask"])
            
            preds.append(output["logits"].detach().cpu())


    # Concatenate all logits, move to CPU, convert to numpy, and squeeze
    all_preds = torch.cat(preds, dim=0).numpy().squeeze()
    np.savez(save_preds_path, preds=all_preds, start_pos=data["start_pos"], end_pos=data["end_pos"],
             chrom=data["chrom"])

def plot(out_fig_path, agg="mean"):
    from matplotlib import colormaps
    # Load sel.csv and bed
    bed_df = pd.read_csv("SEL/bed.bed", sep="\t", header=None)
    bed_df.columns = ["Chromosome", "Start", "End"]
    sel_df = pd.read_csv("SEL/sel.csv")
    
    populations = sel_df["Population"].unique()
    
    colors = colormaps['tab10']
    pop_colors = {pop: colors(i) for i, pop in enumerate(populations)}

    # For each region in bed, plot predictions for each population
    n_regions = len(bed_df)
    fig, axs = plt.subplots(n_regions, 1, figsize=(12, 4 * n_regions), sharex=False)
    if n_regions == 1:
        axs = [axs]
    for idx, row in bed_df.iterrows():
        region_start = row["Start"]
        region_end = row["End"]
        ax = axs[idx]
        for i, pop in enumerate(populations):
            pred_path = f"SEL/{pop}_ftbin_preds.npz"
            data = np.load(pred_path)
            preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
            start_pos = data["start_pos"]
            end_pos = data["end_pos"]
            # Smooth predictions
            window = 15
            if window > 0:
                smooth_preds = np.zeros_like(preds)
                for j in range(len(preds)):
                    left = max(0, j - window // 2)
                    right = min(len(preds), j + window // 2 + 1)
                    smooth_preds[j] = np.mean(preds[left:right])
            else:
                smooth_preds = preds
            # Find indices in predictions overlapping this region
            mask = (start_pos >= region_start) & (end_pos <= region_end)
            region_pos = start_pos[mask]
            region_preds = smooth_preds[mask]
            ax.scatter(region_pos, region_preds, alpha=1, color=colors(i), label=pop)
        ax.set_title(f"{row['Chromosome']}:{region_start}-{region_end}")
        ax.set_xlabel('Position (bp)')
        ax.set_ylabel('p(selection)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
        # Highlight sel region
        sel_mask = (sel_df["Start"] >= region_start) & (sel_df["End"] <= region_end)
        for _, sel_row in sel_df[sel_mask].iterrows():
            ax.axvspan(sel_row["Start"], sel_row["End"], color=pop_colors[sel_row["Population"]], 
                       alpha=0.2, label=f'Selection region, {sel_row["Population"]}')
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')

def plot_manhattan(preds_path_stub, out_fig_path, populations=("CEU", "CHB", "YRI"), window=0):
    """Plot genome-wide predictions in a Manhattan plot style for each population.
    - out_fig_path: path to save the figure
    - populations: iterable of population codes matching saved npz files
    - window: optional smoothing window size (moving average). 0/1 disables smoothing
    """
    # Determine chromosome order and cumulative offsets from the first population
    first_pop = populations[0]
    d0 = np.load(preds_path_stub.format(pop=first_pop))
    chrom0 = d0["chrom"]
    end0 = d0["end_pos"]
    chroms = sorted(np.unique(chrom0))

    # Chromosome lengths and cumulative offsets
    lengths = {c: end0[chrom0 == c].max() for c in chroms}
    offsets = {}
    xticks = []
    xticklabels = []
    run = 0
    for c in chroms:
        offsets[c] = run
        xticks.append(run + lengths[c] / 2)
        xticklabels.append(c)
        run += lengths[c]

    # Alternating colors per chromosome
    colors = plt.colormaps.get("tab20")

    # One subplot per population
    n_rows = len(populations)
    fig, axs = plt.subplots(n_rows, 1, figsize=(20, 3 * n_rows), sharex=True, layout="constrained")
    if n_rows == 1:
        axs = [axs]

    # Load selection regions
    sel_df = pd.read_csv("SEL/sel.csv")

    for i, (ax, pop) in enumerate(zip(axs, populations)):
        data = np.load(preds_path_stub.format(pop=pop))
        logits = torch.tensor(data["preds"])  # [N, 2]
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        if isinstance(window, int) and window > 1:
            kernel = np.ones(window, dtype=float) / window
            probs = np.convolve(probs, kernel, mode="same")
        chrom = data["chrom"]
        starts = data["start_pos"]

        for c in chroms:
            mask = (chrom == c)
            if not np.any(mask):
                continue
            x = starts[mask] + offsets[c]
            y = probs[mask]
            ax.scatter(x, y, s=5, color=colors((c - 1) % 2 + i * 2), alpha=0.7, linewidths=0, rasterized=True)

        # Overlay known selected regions for this population
        added_label = False
        sub = sel_df[sel_df["Population"] == pop]
        for _, r in sub.iterrows():
            c_raw = int(r["Chromosome"].replace("chr", ""))
            x0 = offsets[c_raw] + float(r["Start"])
            x1 = offsets[c_raw] + float(r["End"])
            ax.axvspan(x0, x1, color="purple", # colors(i * 2), 
                       alpha=0.4,
                       label=("Selection region" if not added_label else None))
            added_label=True

        ax.set_ylim(0, 1)
        ax.set_ylabel("p(selection)")
        ax.set_title(f"{pop}")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    axs[0].legend(loc="upper right" )
    axs[-1].set_xticks(xticks)
    axs[-1].set_xticklabels(xticklabels)
    axs[-1].set_xlabel("Chromosome")

    plt.savefig(out_fig_path, dpi=300)
    plt.close(fig)

def plot_chr2_lct(preds_path_stub, out_fig_path, populations=("CEU", "CHB", "YRI"), window=15):
    """Plot predictions for chromosome 2, focusing on the LCT region (136545420..136594754)."""
    colors = plt.colormaps.get('tab10')
    
    # One subplot per population
    n_rows = len(populations)
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axs = [axs]
    
    region_start = 136545420
    region_end = 136594754
    
    for i, (ax, pop) in enumerate(zip(axs, populations)):
        data = np.load(preds_path_stub.format(pop=pop))
        logits = torch.tensor(data["preds"])
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        if isinstance(window, int) and window > 1:
            kernel = np.ones(window, dtype=float) / window
            probs = np.convolve(probs, kernel, mode="same")
        chrom = data["chrom"]
        starts = data["start_pos"]
        
        # Filter for chromosome 2 and the region
        mask = (chrom == 2) # & (starts >= region_start) & (ends <= region_end)
        if np.any(mask):
            x = starts[mask]
            y = probs[mask]
            ax.scatter(x, y, s=5, color=colors(i), alpha=0.7, linewidths=0, rasterized=True)
        
            ax.axvspan(region_start, region_end, color="purple", alpha=0.4,
                        label=("LCT"))
        
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(selection)")
        ax.set_title(f"{pop} - chr2")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("Position (bp)")
    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    model = sys.argv[1]
    if model == "ft":
        path = "models/ft_sel_bin_pan2/"
        preds = "SEL/ftbinpan2_preds_{pop}.npz"
        output = "SEL/ftbinpan2_"
    elif model == "lp":
        path = "models/lp_sel_bin_pan2/"
        preds = "SEL/lpbinpan2_preds_{pop}.npz"
        output = "SEL/lpbinpan2_"
    elif model == "lpft":
        path = "models/lpft_selbin_pan/checkpoint-500"
        preds = "SEL/lpftbinpan_preds_{pop}.npz"
        output = "SEL/lpftbinpan_"
    elif model == "anc":
        path = "models/lp_ancient_x/checkpoint-4600"
        preds = "ANC/preds_{pop}.npz"
        output = "ANC/"

    pops = ["CEU"]
    # pops = ["CEU", "CHB", "YRI"]
    # for pop in pops:
    #     sweep(f"SEL/tokenized_{pop}", path, preds.format(pop=pop))

    # plot(".png", agg="mean")
    plot_manhattan(preds, output + "manhattan.png", populations=pops, window=15)
    plot_chr2_lct(preds, output + "lct.png", populations=pops)