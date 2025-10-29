from time import sleep
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import HapbertaForColumnClassification
from collators import HaploSimpleDataCollator
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import requests

def sweep(dataset, model, save_preds_path=None):
    data = load_from_disk(dataset)

    model = HapbertaForColumnClassification.from_pretrained(
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
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    preds = []

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            output = model(batch["input_ids"], 
                            batch["distances"], 
                            batch["attention_mask"])

            pl = min(len(data[i]["positions"]), 514)
            outs = output["logits"][0].detach().flatten()[:pl].cpu()
            # outs = output["logits"].detach().cpu()
            # print(outs.shape)
            # assert l == outs.shape[0], (l, outs.shape)
            # tqdm.write(f"{outs}")
            preds.append(outs)

    all_preds = torch.cat(preds).numpy().squeeze()

    pos = np.concatenate([x[:min(len(x), 514)] for x in data["positions"]])
    chrom = np.concatenate([np.full(min(len(r),514), c, dtype=np.int8) for c, r in zip(data["chrom"], data["positions"])])

    np.savez(save_preds_path, preds=all_preds, chrom=chrom, positions=pos)


def plot_manhattan(preds_path_stub, out_fig_path):
    """Plot genome-wide predictions in a Manhattan plot style for each population.
    - out_fig_path: path to save the figure
    - populations: iterable of population codes matching saved npz files
    - window: optional smoothing window size (moving average). 0/1 disables smoothing
    """
    # Determine chromosome order and cumulative offsets from the first population
    df = pd.read_csv("ANC/Selection_Summary_Statistics_01OCT2025.tsv", comment="#", sep="\t")
    ds = np.load(preds_path_stub)
    chrom = ds["chrom"]
    positions = ds["positions"]
    preds = ds["preds"]
    preds = np.abs(preds)

    df_preds = pd.DataFrame({'chrom': chrom, 'positions': positions, 'preds': preds})
    df_collapsed = df_preds.groupby(['chrom', 'positions']).agg({'preds': 'mean'}).reset_index()
    chrom = df_collapsed['chrom'].values.astype(np.int8)
    positions = df_collapsed['positions'].values
    preds = df_collapsed['preds'].values
    chroms = sorted(np.unique(chrom))

    # smooth preds
    window = 1
    if window > 1:
        preds = np.convolve(preds, np.ones(window)/window, mode='same')

    # Chromosome lengths and cumulative offsets
    lengths = {c: positions[chrom == c].max() for c in chroms}
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
    fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex=True, layout="constrained")
    
    posmask = np.isin(positions, df["POS"].values)
    for c in chroms:
        mask = (chrom == c) & posmask
        if not np.any(mask):
            continue
        x = positions[mask] + offsets[c]
        y = preds[mask]
        ax.scatter(x, y, s=5, color=colors((c - 1) % 2), alpha=0.7, linewidths=0, rasterized=True)

    # ax.set_ylim(0, 5)
    ax.set_ylabel("pred. s")
    # ax.set_title("")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Chromosome")

    plt.savefig(out_fig_path, dpi=300)
    plt.close(fig)

def high_preds_query(preds_path_stub, out_preds_path):
    """"""
    # Determine chromosome order and cumulative offsets from the first population
    ds = np.load(preds_path_stub)
    chrom = ds["chrom"]
    positions = ds["positions"]
    preds = ds["preds"]
    preds = np.abs(preds)

    df_preds = pd.DataFrame({'chrom': chrom, 'positions': positions, 'preds': preds})
    df_collapsed = df_preds.groupby(['chrom', 'positions']).agg({'preds': 'mean'}).reset_index()
    chrom = df_collapsed['chrom'].values.astype(np.int8)
    positions = df_collapsed['positions'].values
    preds = df_collapsed['preds'].values

    # Get top 500 predictions
    top_idx = np.argsort(preds)[-250:]
    top_chrom = chrom[top_idx]
    top_pos = positions[top_idx]
    top_preds = preds[top_idx]


    results = []
    for c, p, pred in tqdm(zip(top_chrom, top_pos, top_preds)):
        # Ensembl REST API expects chromosome as string, positions as int
        server = "https://grch37.rest.ensembl.org"
        ext = f"/overlap/region/human/{c}:{int(p)}-{int(p)}?feature=gene"
        r = requests.get(server + ext, headers={ "Content-Type" : "application/json"})
        gene_name, name = None, None
        if r.ok and r.json():
            # Take the first gene hit, or join all gene names
            genes = [g.get("gene_id", "") for g in r.json()]
            gene_name = ";".join(genes)
            names = [g.get("external_name", "") for g in r.json()]
            name = ";".join(names)
        results.append({"chrom": c, "position": p, "pred": pred, "gene": gene_name, "name": name})
        sleep(0.08)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_preds_path, index=False)


def test_evens(dataset, model):
    data = load_from_disk(dataset)
    data = data.filter(lambda ex: ex["chrom"] % 2 == 0)

    model = HapbertaForColumnClassification.from_pretrained(
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
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    preds = []

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            output = model(batch["input_ids"], 
                            batch["distances"], 
                            batch["attention_mask"])

            pl = min(len(data[i]["positions"]), 514)
            outs = output["logits"][0].detach().flatten()[:pl].cpu()
            preds.append(outs)

    all_preds = torch.cat(preds).numpy().squeeze()
    pos = np.concatenate([x[:min(len(x), 514)] for x in data["positions"]])
    chrom = np.concatenate([np.full(min(len(r),514), c, dtype=np.int8) for c, r in zip(data["chrom"], data["positions"])])
    
    df_preds = pd.DataFrame({'CHROM': chrom, 'POS': pos, 'PREDS': all_preds})
    df_collapsed = df_preds.groupby(['CHROM', 'POS']).agg({'PREDS': 'mean'})

    df = pd.read_csv("ANC/Selection_Summary_Statistics_01OCT2025.tsv", comment="#", sep="\t")

    # Assuming df has columns 'CHROM', 'POS', 'S'
    df = df.drop_duplicates(subset=["CHROM", "POS"])
    df.set_index(['CHROM', 'POS'], inplace=True)
    
    # Get labels, handling missing values
    labels = df.reindex(df_collapsed.index)['S']
    mask = ~labels.isna()

    # Filter predictions and labels
    filtered_preds = df_collapsed["PREDS"].values[mask]
    filtered_labels = labels[mask]

    # Calculate metrics
    r2 = r2_score(filtered_labels, filtered_preds)
    mse = mean_squared_error(filtered_labels, filtered_preds)
    mae = mean_absolute_error(filtered_labels, filtered_preds)

    print(f"R^2: {r2}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

if __name__ == "__main__":
    path = "models/lp_ancient_x/"
    preds = "ANC/preds_lp_CEU.npz"
    output = "ANC/"

    # test_evens("ANC/tokenized_CEU", path)

    pops = ["CEU"]
    pops = ["CEU", "CHB", "YRI"]
    for pop in pops:
        sweep(f"ANC/tokenized_{pop}", path, preds)
        high_preds_query(preds, output + "_preds.csv")
        plot_manhattan(preds, output + f"{pop}_manhattan_lp.png")
