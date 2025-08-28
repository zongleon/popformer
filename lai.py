import numpy as np
import torch
from models import HapbertaForSequenceClassification
from collators import HaploSimpleNormalDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

def sweep():
    data = load_from_disk(f"LAI/LAI_CEU_test")

    model = HapbertaForSequenceClassification.from_pretrained(
        "models/hapberta2d_pop/checkpoint-500",
        num_labels=3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    collator = HaploSimpleNormalDataCollator(subsample=32, label_dtype=torch.long)

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
            # pred = output["logits"].cpu().numpy().squeeze()
            
            # instead, get softmax logits for class 1
            pred = output["logits"].argmax(dim=-1).cpu().numpy().squeeze()
            # print(pred)
            preds.append(pred)

    preds = np.concatenate(preds[:-1], axis=0)
    np.savez(f"LAI/LAI_CEU_preds_test.npz", preds=preds, pos=data["pos"])

def aggregate_predictions():
    """Aggregate overlapping window predictions to get class proportions at each position"""
    data = np.load(f"LAI/LAI_CEU_preds_test.npz")
    preds = data["preds"]
    pos = data["pos"]
    
    # pos is (n_windows, 2) with [chrom, start_pos]
    chrom = pos[:, 0]
    start_pos = pos[:, 1]
    window_size = 512
    
    # Create a dictionary to store predictions for each position
    position_preds = {}
    
    for i in range(len(preds)):
        # Each window covers positions from start_pos to start_pos + window_size - 1
        window_start = start_pos[i]
        window_chrom = chrom[i]
        pred_class = preds[i]
        
        # Add this prediction to all positions in the window
        for pos_offset in range(window_size):
            position = (window_chrom, window_start + pos_offset)
            if position not in position_preds:
                position_preds[position] = []
            position_preds[position].append(pred_class)
    
    # Calculate proportions for each position
    positions = sorted(position_preds.keys())
    class_proportions = {0: [], 1: [], 2: []}
    position_coords = []
    
    for pos in positions:
        preds_at_pos = position_preds[pos]
        total_preds = len(preds_at_pos)
        
        # Count each class
        class_counts = {0: 0, 1: 0, 2: 0}
        for pred in preds_at_pos:
            class_counts[pred] += 1
        
        # Calculate proportions
        for class_id in [0, 1, 2]:
            proportion = class_counts[class_id] / total_preds
            class_proportions[class_id].append(proportion)
        
        position_coords.append(pos)
    
    # Save aggregated results
    np.savez(f"LAI/LAI_CEU_aggregated_preds.npz", 
             positions=np.array(position_coords),
             class0_prop=np.array(class_proportions[0]),
             class1_prop=np.array(class_proportions[1]),
             class2_prop=np.array(class_proportions[2]))
    
    return position_coords, class_proportions

def plot():
    # Load aggregated data
    try:
        data = np.load(f"LAI/LAI_CEU_aggregated_preds.npz")
        positions = data["positions"]
        class0_prop = data["class0_prop"]
        class1_prop = data["class1_prop"]
        class2_prop = data["class2_prop"]
    except FileNotFoundError:
        print("Aggregated data not found, computing...")
        positions, class_proportions = aggregate_predictions()
        class0_prop = np.array(class_proportions[0])
        class1_prop = np.array(class_proportions[1])
        class2_prop = np.array(class_proportions[2])
    
    # Extract chromosome and position for plotting
    chrom = positions[:, 0]
    pos = positions[:, 1]
    
    # Plot class proportions
    plt.figure(figsize=(12, 8))
    plt.plot(pos, class0_prop, label='Class 0', alpha=0.7)
    plt.plot(pos, class1_prop, label='Class 1', alpha=0.7)
    plt.plot(pos, class2_prop, label='Class 2', alpha=0.7)
    plt.xlabel('Genomic Position')
    plt.ylabel('Class Proportion')
    plt.title('Local Ancestry Inference - Class Proportions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('LAI/LAI_CEU_class_proportions.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    sweep()
    aggregate_predictions()
    plot()