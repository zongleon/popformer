import sys
import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

def sweep(dataset, model, save_preds_path=None):
    data = load_from_disk(dataset)

    model = HapbertaForSequenceClassification.from_pretrained(
        model,
        torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleDataCollator(subsample=32)

    batch_size = 16
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch_samples = [data[j] for j in range(i, min(i + batch_size, len(data)))]
            # if len(batch_samples) != batch_size:
            #     continue
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
    np.savez(save_preds_path, preds=preds, start_pos=data["start_pos"], end_pos=data["end_pos"],
             chrom=data["chrom"])

def plot(save_preds_path, out_fig_path, mode="pop", agg="mean"):
    assert mode in ["pop", "sel"], "Only can plot pop classification or selection preds"

    data = np.load(save_preds_path)
    # preds = torch.softmax(torch.tensor(data["preds"]), dim=-1)[:, 1].numpy()
    preds = data["preds"]
    start_pos = data["start_pos"]
    end_pos = data["end_pos"]
    
    if mode == "sel":
        d = 1
    else:
        d = preds.shape[1]
    p = np.zeros((end_pos[-1] - start_pos[0], d))
    counts = np.zeros_like(p)
    for i in range(len(preds)):
        s = start_pos[i] - start_pos[0]
        e = end_pos[i] - start_pos[0]
        if agg == "max":
            for pos in range(s, e):
                if p[pos] < preds[i]:
                    p[pos] = preds[i]
        elif agg == "mean":
            # sum
            p[s:e] += preds[i]
            counts[s:e] += 1
    # Avoid division by zero
    if agg == "mean":
        counts[counts == 0] = 1
        p /= counts


    pos = np.arange(start_pos[0], end_pos[-1])
    if "reg" not in save_preds_path: 
        ps = torch.softmax(torch.tensor(p), dim=-1).numpy()

        # Only plot positions where all output logits were not zero
        ps = np.ma.masked_where((p == 0), ps)
    else:
        ps = p

    # Create figure with nice size
    plt.figure(figsize=(12, 6))
    
    if mode == "pop":
        # Stacked area chart for population probabilities
        for i in range(3):
            plt.plot(pos, ps[:, i], label=["CEU", "CHB", "YRI"][i], alpha=0.8)
        lbl = "Probability of population"
        tit = "Predicting population label"
        plt.legend(title="Population", loc="upper right")
    
    elif mode == "sel":
        if "reg" in save_preds_path:
            plt.plot(pos, ps)
        else:
            plt.plot(pos, ps[:, 1])
        lbl = "Probability of selection"
        tit = "Predicting selection along real genome"

    # Styling
    plt.xlabel(f'Chromosome {data["chrom"][0]} Position (bp)')
    plt.ylabel(lbl)
    plt.title(tit)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to show positions in a readable format
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    sweep("SEL/tokenized_CEU", "models/hapberta2d_sel/", "SEL/CEU_reg_preds.npz")
    plot("SEL/CEU_reg_preds.npz", "SEL/fig.png", "sel")