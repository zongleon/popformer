import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleNormalDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

def sweep(t: str):
    data = load_from_disk(f"GHIST/ghist_samples_{t}")

    model = HapbertaForSequenceClassification.from_pretrained(
        "models/hapberta2d_sel_binary/checkpoint-1000",
        # num_labels=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleNormalDataCollator(label_dtype=torch.float32)

    batch_size = 16
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
            # pred = output["logits"].cpu().numpy().squeeze()
            
            # instead, get softmax logits for class 1
            pred = output["logits"].softmax(dim=-1)[:, 1].cpu().numpy().squeeze()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    np.savez(f"GHIST/ghist_preds2_{t}.npz", preds=preds, pos=data["pos"])

def plot(t: str):
    data = np.load(f"GHIST/ghist_preds2_{t}.npz")
    preds = data["preds"]
    pos = data["pos"]
    
    # Create figure with nice size
    plt.figure(figsize=(12, 6))
    
    # Plot predictions vs positions
    plt.plot(pos, preds, linewidth=0.8, alpha=0.8)
    
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
    plt.savefig(f'GHIST/selection_coefficients_plot2_{t}.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # t = "singlesweep.growth_bg"
    t = "singlesweep"
    sweep(t)
    plot(t)