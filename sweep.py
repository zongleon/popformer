import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from popformer.models import PopformerForWindowClassification
from popformer.collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def sweep(dataset, model, save_preds_path=None, save_features_path=None, subsample=None):
    data = load_from_disk(dataset)

    model = PopformerForWindowClassification.from_pretrained(
        model,
        torch_dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=(subsample, subsample) if subsample else None,
                                       subsample_type="diverse")

    loader = DataLoader(
        data,
        batch_size=4,
        num_workers=4,
        collate_fn=collator,
    )
    preds = []
    features = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            output = model(batch["input_ids"], 
                            batch["distances"], 
                            batch["attention_mask"],
                            return_hidden_states=save_features_path is not None
                            )
            
            if save_preds_path:
                preds.append(output["logits"].detach().cpu())
            if save_features_path:
                # mean pool them
                features.append(output["hidden_states"].mean(dim=(1,2)).detach().cpu())

    if "positions" in data.column_names:
        start_pos = [p[0] for p in data["positions"]]
        end_pos = [p[-1] for p in data["positions"]]
        chrom = data["chrom"]
    elif "start_pos" in data.column_names and "end_pos" in data.column_names:
        start_pos = data["start_pos"]
        end_pos = data["end_pos"]
        chrom = data["chrom"]
    else:
        start_pos = np.array([])
        end_pos = np.array([])
        chrom = np.array([])

    if save_features_path:
        all_features = torch.cat(features, dim=0).numpy()
        print(f"Saving features of shape {all_features.shape} to {save_features_path}")
        np.savez(save_features_path, features=all_features, start_pos=start_pos, end_pos=end_pos,
                 chrom=chrom)

    if save_preds_path:
        all_preds = torch.cat(preds, dim=0).numpy().squeeze()
        print(f"Saving predictions of shape {all_preds.shape} to {save_preds_path}")
        np.savez(save_preds_path, preds=all_preds, start_pos=start_pos, end_pos=end_pos,
                 chrom=chrom)


def plot(out_fig_path, agg="mean"):
    from matplotlib import colormaps

    # Load sel.csv and bed
    sel_df = pd.read_csv("SEL/sel.csv")
    
    populations = sel_df["Population"].unique()
    
    colors = colormaps['tab10']
    pop_colors = {pop: colors(i) for i, pop in enumerate(populations)}

    # For each region in bed, plot predictions for each population
    n_regions = len(sel_df)
    fig, axs = plt.subplots(n_regions, 1, figsize=(12, 4 * n_regions), sharex=False)
    if n_regions == 1:
        axs = [axs]
    for idx, row in sel_df.iterrows():
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
        # logits = torch.tensor(data["preds"])  # [N, 2]
        # probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        probs = data["preds"][:, 1]
        # probs = data["preds"]  # [N,]
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

        # ax.set_ylim(0, 1)
        ax.set_ylabel("p(selection)")
        ax.set_title(f"{pop}")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    axs[0].legend(loc="upper right" )
    axs[-1].set_xticks(xticks)
    axs[-1].set_xticklabels(xticklabels)
    axs[-1].set_xlabel("Chromosome")

    plt.savefig(out_fig_path, dpi=300)
    plt.close(fig)


def plot_region(preds_path, out_fig_path, window=0, label_df=None):
    data = np.load(preds_path)
    # preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
    preds = data["preds"][:, 1]
    ylbl = "pred. probability of selection"
    
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]

    # Smooth predictions by averaging over surrounding predictions

    if isinstance(window, int) and window > 1:
        kernel = np.ones(window, dtype=float) / window
        preds = np.convolve(preds, kernel, mode="same")

    plt.figure(figsize=(12, 6), layout="constrained")
    plt.scatter(start_pos, preds, alpha=0.8)

    if label_df is not None:
        for _, r in label_df.iterrows():
            x0 = r["start"]
            x1 = r["end"]
            plt.axvspan(x0, x1, color="purple", alpha=0.4)
        plt.legend(["Predictions", "Selection region"])

    plt.xlabel('Position (bp)')
    plt.ylabel(ylbl)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.savefig(out_fig_path, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--save_features', type=str, help='Path for saving features')
    parser.add_argument('--save_logits', type=str, help='Path for saving logits')
    parser.add_argument('--logits_path', type=str, help='Path for loading logits')
    parser.add_argument('--plot_preds', type=str, help='Path to save the Manhattan plot')
    parser.add_argument('--plot_region', type=str, help='Path to save the region plot')
    parser.add_argument('--region_labels', type=str, help='CSV file with region labels for region plot')
    parser.add_argument('--smooth_window', type=int, default=7, help='Smoothing window size')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample size for collator')

    args = parser.parse_args()

    data = args.data
    model = args.model
    preds_path = None

    if args.save_logits or args.save_features:
        preds_path = args.save_logits
        sweep(data, model, args.save_logits, args.save_features, subsample=args.subsample)
    
    if args.plot_preds:
        preds_path = args.logits_path if args.logits_path else preds_path
        if not os.path.exists(preds_path):
            raise ValueError(f"Predictions path {preds_path} does not exist. Run with --save_logits first or specify with --logits_path.")
        plot_manhattan(preds_path, args.plot_preds, populations=("CEU",), window=args.smooth_window)
    
    if args.plot_region:
        preds_path = args.logits_path if args.logits_path else preds_path
        if not os.path.exists(preds_path):
            raise ValueError(f"Predictions path {preds_path} does not exist. Run with --save_logits first or specify with --logits_path.")
        # Load selection regions
        sel_df = pd.read_csv(args.region_labels)
        plot_region(preds_path, args.plot_region, window=args.smooth_window, label_df=sel_df)
