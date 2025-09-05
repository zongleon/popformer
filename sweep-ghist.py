import sys
import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

def sweep(t: str):
    data = load_from_disk(f"GHIST/ghist_samples_{t}")

    model = HapbertaForSequenceClassification.from_pretrained(
        "models/hapberta2d_sel_binary",
        torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()
    # compiled_model = torch.compile(model, fullgraph=True)

    collator = HaploSimpleDataCollator(subsample=32)

    batch_size = 8
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_samples = [data[j] for j in range(i, min(i + batch_size, len(data)))]
            batch = collator(batch_samples)
            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            output = model(batch["input_ids"], batch["distances"], batch["attention_mask"])
            
            pred = output["logits"].to(torch.float16).cpu().numpy().squeeze()
            preds.append(pred)

    # print(preds[-1])
    preds = np.concatenate(preds, axis=0)
    np.savez(f"GHIST/ghist_preds3_{t}.npz", preds=preds, start_pos=data["start_pos"], end_pos=data["end_pos"])

def plot(t: str):
    data = np.load(f"GHIST/ghist_preds3_{t}.npz")
    preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
    # preds = data["preds"]
    # print(preds.shape)
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]
    
    p = np.zeros((end_pos[-1] - start_pos[0]))
    counts = np.zeros_like(p)
    for i in range(len(preds)):
        s = start_pos[i] - start_pos[0]
        e = end_pos[i] - start_pos[0]
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
    plt.xlabel('Chromosome Position (bp)')
    plt.ylabel('Predicted Selection Coefficient')
    plt.title('Predicted Selection Coefficients Across Chromosome')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to show positions in a readable format
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'GHIST/selection_coefficients_plot3_{t}.png', dpi=300, bbox_inches='tight')


def select(t: str, threshold: float = 0.2):
    data = np.load(f"GHIST/ghist_preds2_{t}.npz")
    preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]

    # Compute average prediction per position as in plot()
    p = np.zeros((end_pos[-1] - start_pos[0]))
    counts = np.zeros_like(p)
    for i in range(len(preds)):
        s = start_pos[i] - start_pos[0]
        e = end_pos[i] - start_pos[0]
        p[s:e] += preds[i]
        counts[s:e] += 1
    counts[counts == 0] = 1
    avg_pred = p / counts
    pos = np.arange(start_pos[0], end_pos[-1])

    # Find contiguous ranges where avg_pred > threshold
    above = avg_pred > threshold
    ranges = []
    in_range = False
    for i, flag in enumerate(above):
        if flag and not in_range:
            range_start = pos[i]
            in_range = True
        elif not flag and in_range:
            range_end = pos[i-1] + 1  # end is exclusive
            ranges.append((range_start, range_end))
            in_range = False
    if in_range:
        ranges.append((range_start, pos[-1]+1))

    # Output the ranges
    print(f"(avg_pred > {threshold})")
    for r in ranges:
        print(f"21\t{r[0]}\t{r[1]}".expandtabs(4))


if __name__ == "__main__":
    t = sys.argv[1]
    sweep(t)
    plot(t)
    # select(t, 0.2)