import sys
import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def sweep(t: str, model: str, name: str):
    data = load_from_disk(f"GHIST/samples_{t}")

    model = HapbertaForSequenceClassification.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        num_labels=2 if "bin" in name else 1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()
    # model.compile()

    collator = HaploSimpleDataCollator(subsample=(50, 50), pad_batch=True)

    batch_size = 16
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_samples = [data[j] for j in range(i, min(i + batch_size, len(data)))]
            batch = collator(batch_samples)
            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device=device)
            
            output = model(batch["input_ids"], batch["distances"], batch["attention_mask"])
            preds.append(output["logits"])

    # Concatenate all logits, move to CPU, convert to numpy, and squeeze
    all_preds = torch.cat(preds, dim=0).to(torch.float16).cpu().numpy().squeeze()
    np.savez(f"GHIST/{name}_{t}.npz", preds=all_preds, start_pos=data["start_pos"], end_pos=data["end_pos"])


def sweep_snpdensity(t: str):
    data = load_from_disk(f"GHIST/ghist_samples_{t}")

    collator = HaploSimpleDataCollator(subsample=32, pad_batch=True)

    preds = []

    with torch.no_grad():
        for i in tqdm(range(len(data))):
            batch_samples = [data[i]]
            batch = collator(batch_samples)
            preds.append(batch["input_ids"].shape[2])

    preds = np.array(preds)
    np.savez(f"GHIST/snpdens_{t}.npz", preds=preds, start_pos=data["start_pos"], end_pos=data["end_pos"])

def plot_smooth(t: str, name: str):
    data = np.load(f"GHIST/{name}_{t}.npz")
    if "snpdens" in name:
        preds = data["preds"]
        ylbl = "num SNPs in window"
    elif "reg" in name:
        preds = data["preds"]
        ylbl = "pred. selection coeff."
    else:
        preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        # preds = np.log(preds)
        ylbl = "pred. probability of selection"
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]

    # Smooth predictions by averaging over surrounding predictions
    window = 7
    smooth_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        left = max(0, i - window // 2)
        right = min(len(preds), i + window // 2 + 1)
        smooth_preds[i] = np.mean(preds[left:right])
    pos = start_pos

    plt.figure(figsize=(12, 6))
    plt.plot(pos, smooth_preds, linewidth=0.8, alpha=0.8)
    plt.xlabel('Position (bp)')
    plt.ylabel(ylbl)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(f'GHIST/{name}_{t}_smooth.png', dpi=300, bbox_inches='tight')


def plot_combine(name: str):
    ts = ["singlesweep", "singlesweep.growth_bg", "multisweep", "multisweep.growth_bg"]
    ts = ["rl" + t + "_100000" for t in ts]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for idx, t in enumerate(ts):
        data = np.load(f"GHIST/{name}_{t}.npz")
        if "snpdens" in name:
            preds = data["preds"]
            ylbl = "num SNPs in window"
        elif "reg" in name:
            preds = data["preds"]
            ylbl = "pred. selection coeff."
        else:
            preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
            # preds = np.log(preds)
            # preds = data["preds"][:, 1]
            ylbl = "pred. probability of selection"
        start_pos = data["start_pos"]

        # Smooth predictions by averaging over surrounding predictions
        window = 9
        smooth_preds = np.zeros_like(preds)
        for i in range(len(preds)):
            left = max(0, i - window // 2)
            right = min(len(preds), i + window // 2 + 1)
            smooth_preds[i] = np.mean(preds[left:right])

        # Scale predictions to [0, 1]
        # smooth_preds = (smooth_preds - np.min(smooth_preds)) / (np.max(smooth_preds) - np.min(smooth_preds) + 1e-8)

        # peaks
        peaks, _ = find_peaks(smooth_preds, width=None, distance=20, prominence=None)

        # Subset top 15 peaks by prominence
        if len(peaks) > 0:
            top = 1
            if "multisweep" in t:
                top = 6
            if "multisweep.growth_bg" in t:
                top = 15
            prominences = smooth_preds[peaks]
            top_indices = np.argsort(prominences)[-top:]
            top_peaks = peaks[top_indices]
            # Sort top_peaks by position for plotting
            top_peaks = np.sort(top_peaks)

            # Print out a BED file for the top peaks
            bed_lines = []
            for peak_idx in top_peaks:
                pos = start_pos[peak_idx]
                # BED format: chrom, start, end, name, score
                bed_lines.append(f"21\t{pos-100000}\t{pos+300000}")

            bed_file = f"GHIST/submit/{name}_{t}_3.bed"
            with open(bed_file, "w") as f:
                for line in bed_lines:
                    f.write(line + "\n")
        else:
            top_peaks = np.array([])

        # Plot only top 15 peaks
        ax = axs[idx // 2, idx % 2]
        ax.plot(start_pos, smooth_preds, linewidth=0.8, alpha=0.8)
        ax.plot(start_pos[top_peaks], smooth_preds[top_peaks], "x")
        ax.set_title(t.rstrip("_50000"))
        ax.set_xlabel('Position (bp)')
        ax.set_ylabel(ylbl)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(f'GHIST/{name}_combine_smooth.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    t = sys.argv[1]
    name = sys.argv[2]

    model = None
    if len(sys.argv) > 3:
        model = sys.argv[3]

    if model is not None:
        if model != "nopred":
            sweep(t, model, name)
    else:
        plot_combine(name)
        raise SystemExit
    # sweep_snpdensity(t)
    # plot(t, name)
    # plot2(t, name)
    plot_smooth(t, name)
    # select(t, 0.2)