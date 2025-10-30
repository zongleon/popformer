import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_smooth(preds_path: str, output: str):
    data = np.load(preds_path)
    if "snpdens" in preds_path:
        preds = data["preds"]
        ylbl = "num SNPs in window"
    elif "selreg" in preds_path:
        preds = data["preds"]
        ylbl = "pred. selection coeff."
    else:
        preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        # preds = np.log(preds)
        ylbl = "pred. probability of selection"
        
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]

    # Smooth predictions by averaging over surrounding predictions
    window = 5
    smooth_preds = np.zeros_like(preds)
    for i in range(len(preds)):
        left = max(0, i - window // 2)
        right = min(len(preds), i + window // 2 + 1)
        smooth_preds[i] = np.mean(preds[left:right])
    pos = start_pos

    plt.figure(figsize=(12, 6), layout="constrained")
    plt.scatter(pos, smooth_preds, alpha=0.8)
    plt.xlabel('Position (bp)')
    plt.ylabel(ylbl)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.savefig(output, dpi=300)


def plot_combine(preds_path: str, output: str):
    ts = ["singlesweep", "singlesweep.growth_bg", "multisweep", "multisweep.growth_bg"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
    for idx, t in enumerate(ts):
        data = np.load(preds_path.format(t=t))
        # preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        preds = data["preds"][:, 1]
        ylbl = "pred. probability of selection"
        start_pos = data["start_pos"]

        # Smooth predictions by averaging over surrounding predictions
        window = 5
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

            bed_file = "GHIST/submit/temp.bed" 
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

    plt.savefig(output.format(t="smooth"), dpi=300)

    
if __name__ == "__main__":
    preds_stub = sys.argv[1]
    fig_stub = sys.argv[2]
    # for t in ["singlesweep", "singlesweep.growth_bg", "multisweep", "multisweep.growth_bg"]:
    #     plot_smooth(preds_stub.format(t=t), fig_stub.format(t=t))
        
    plot_combine(preds_stub, fig_stub)