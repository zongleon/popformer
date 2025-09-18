import sys
import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    collator = HaploSimpleDataCollator(subsample=32, pad_batch=True)

    batch_size = 32
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


def plot(t: str, name: str):
    data = np.load(f"GHIST/{name}_{t}.npz")
    if "snpdens" in name:
        preds = data["preds"]
        ylbl = "num SNPs in window"
    elif "reg" in name:
        preds = data["preds"] / 100
        ylbl = "pred. selection coeff."
    else:
        preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        # preds = data["preds"][:, 1]
        ylbl = "pred. probability of selection"
    # print(preds.shape)
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]
    
    p = np.zeros((end_pos[-1] - start_pos[0]))
    counts = np.zeros_like(p)
    for i in range(len(preds)):
        s = start_pos[i] - start_pos[0]
        e = end_pos[i] - start_pos[0]
        # print(s, e, preds[i])
        p[s:e] += preds[i]
        counts[s:e] += 1
    # Avoid division by zero
    counts[counts == 0] = 1
    p /= counts
    pos = np.arange(start_pos[0], end_pos[-1])

    # Create figure with nice size
    plt.figure(figsize=(12, 6))
    
    # Plot predictions vs positions
    plt.plot(pos, p, linewidth=0.8, alpha=0.8)
    
    # Styling
    plt.xlabel('Position (bp)')
    plt.ylabel(ylbl)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to show positions in a readable format
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'GHIST/{name}_{t}_line.png', dpi=300, bbox_inches='tight')


def plot2(t: str, name: str):
    data = np.load(f"GHIST/{name}_{t}.npz")
    if "reg" in name:
        preds = data["preds"] / 100
        ylbl = "pred. selection coeff."
    else:
        preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        # preds = data["preds"][:, 1]
        ylbl = "pred. probability of selection"
    # print(preds.shape)
    start_pos = data["start_pos"][:preds.shape[0]]
    end_pos = data["end_pos"]
    
    print(start_pos.shape, preds.shape)
    # Create figure with nice size
    plt.figure(figsize=(12, 6))
    
    # Plot predictions vs positions
    plt.scatter(start_pos, preds, linewidth=0.8, alpha=0.8)
    
    # Styling
    plt.xlabel('Position (bp)')
    plt.ylabel(ylbl)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to show positions in a readable format
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'GHIST/{name}_{t}_rawscatter.png', dpi=300, bbox_inches='tight')


def plot_smooth(t: str, name: str):
    data = np.load(f"GHIST/{name}_{t}.npz")
    if "snpdens" in name:
        preds = data["preds"]
        ylbl = "num SNPs in window"
    elif "reg" in name:
        preds = data["preds"] / 100
        ylbl = "pred. selection coeff."
    else:
        preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
        ylbl = "pred. probability of selection"
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]

    # Smooth predictions by averaging over surrounding 9 predictions
    window = 9
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
    ts = [t + "_50000" for t in ts]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for idx, t in enumerate(ts):
        data = np.load(f"GHIST/{name}_{t}.npz")
        if "snpdens" in name:
            preds = data["preds"]
            ylbl = "num SNPs in window"
        elif "reg" in name:
            preds = data["preds"] / 100
            ylbl = "pred. selection coeff."
        else:
            preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
            ylbl = "pred. probability of selection"
        start_pos = data["start_pos"]

        # Smooth predictions by averaging over surrounding 9 predictions
        window = 15
        smooth_preds = np.zeros_like(preds)
        for i in range(len(preds)):
            left = max(0, i - window // 2)
            right = min(len(preds), i + window // 2 + 1)
            smooth_preds[i] = np.mean(preds[left:right])

        # Scale predictions to [0, 1]
        # smooth_preds = (smooth_preds - np.min(smooth_preds)) / (np.max(smooth_preds) - np.min(smooth_preds) + 1e-8)

        ax = axs[idx // 2, idx % 2]
        ax.plot(start_pos, smooth_preds, linewidth=0.8, alpha=0.8)
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
        sweep(t, model, name)
    else:
        plot_combine(name)
    # sweep_snpdensity(t)
    # plot(t, name)
    # plot2(t, name)
    plot_smooth(t, name)
    # select(t, 0.2)