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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve

def test(dataset, model, save_preds_path=None):
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
    np.save(save_preds_path, all_preds)

def acc(preds_path):
    preds = np.load(preds_path)
    preds = torch.softmax(torch.tensor(preds), dim=-1)[:, 1].numpy()
    np.savetxt("tp.out", preds, fmt="%.3f")
    # columns ["dataset", "label"]
    # dataset has items like D1, D2, D3...
    # label has binary classes 0/1
    metadata = pd.read_csv("FASTER_NN/fasternn_meta.csv")

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flat
    for i, dataset_name in enumerate(metadata["dataset"].unique()):
        idx = metadata["dataset"] == dataset_name
        dataset_labels = metadata.loc[idx, "label"].values
        dataset_preds = preds[idx]

        dataset_auc = roc_auc_score(dataset_labels, dataset_preds)

        # Plot ROC curve for each dataset
        fpr, tpr, thresholds = roc_curve(dataset_labels, dataset_preds)
        axs[i].plot(fpr, tpr, label=f"{dataset_name} (AUC={dataset_auc:.2f})")

        accuracies = [
            accuracy_score(dataset_labels, dataset_preds >= thr) for thr in thresholds
        ]
        best_idx = max(range(len(accuracies)), key=accuracies.__getitem__)

        dataset_acc = accuracy_score(dataset_labels, dataset_preds > thresholds[best_idx])
        print(f"Accuracy @ thr={thresholds[best_idx]:.3f}, {dataset_name}: {dataset_acc:.4f}")
        print(f"AUC-ROC,  {dataset_name}: {dataset_auc:.4f}")


    plt.savefig("figs/fasternn_rocs.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    model = sys.argv[1]
    # path = "models/ft_selbin3/checkpoint-1100"
    path = model
    preds = "FASTER_NN/preds.npy"
    # output = "FASTER_NN/ftbinpan2_"
    # preds = "FASTER_NN/lpbinpan2_preds.npy"
    
    test("FASTER_NN/tokenized_majmin512", path, preds)
    acc(preds)
    
