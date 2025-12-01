import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

sys.path.append(".")
from popformer.dataset import find_nonzero_block_cols

FILE = "{pop}_{typ}{seed}.trees"
MAX_HAPS = 256
MAX_SNPS = 2048

def main(path, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(path, "metadata.csv"))
    cols = df.columns.tolist()

    all_mats = [np.load(os.path.join(path, "matrices.npy"))]
    all_dists = [np.load(os.path.join(path, "distances.npy"))]

    for t, sel in zip(
        ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
        [0, 0.1, 0.01, 0.05, 0.025],
    ):
        for pop in ["CEU", "CHB", "YRI"]:
            print(f"Processing {pop} {t}")
            lm = np.load(f"data/matrices/{pop}/matrices_regions_{pop}_{t}.npy")
            ld = np.load(f"data/matrices/{pop}/distances_regions_{pop}_{t}.npy")
            
            n = lm.shape[0]
            n_haps = lm.shape[2]

            # store results
            matrices = np.full((n, MAX_HAPS, MAX_SNPS), 5, dtype=np.int8)
            distances = np.zeros((n, MAX_SNPS), dtype=np.float16)
            
            for idx, (mat, dist) in enumerate(zip(lm, ld)):
                mat = mat.T

                first, last = find_nonzero_block_cols(mat)
                
                matrices[idx, :n_haps, :(last-first)] = mat[:, first:last]
                distances[idx, :(last-first)] = dist[None, first:last] * 50000

            # metadata
            data = [(seed, 
                     -1 if "neutral" in t else 25000,
                     sel, 0, 0, "old", pop) for seed in range(n)]
            
            # combine!
            df = pd.concat([df, pd.DataFrame(data, columns=cols)])
            all_mats.append(matrices)
            all_dists.append(distances)

    np.save(os.path.join(out, "matrices.npy"), np.concatenate(all_mats))
    np.save(os.path.join(out, "distances.npy"), np.concatenate(all_dists))
    df.to_csv(os.path.join(out, "metadata.csv"), index=False)

def stats(path):
    matrices = np.load(os.path.join(path, "matrices.npy"))
    df = pd.read_csv(os.path.join(path, "metadata.csv"))

    # Compute nonzero block cols for each matrix
    block_lengths = []
    for mat in matrices:
        first, last = find_nonzero_block_cols(mat)
        block_lengths.append(last - first)
    df["block_length"] = block_lengths

    pops = df["pop"].unique()
    sims = df["sim"].unique() if "sim" in df.columns else ["sim"]

    fig, axes = plt.subplots(len(pops), len(sims), figsize=(12, 12), sharex=True, sharey=True)
    for i, pop in enumerate(pops):
        for j, sim in enumerate(sims):
            sub = df[(df["pop"] == pop) & (df["sim"] == sim)]
            bl0 = sub[sub["coeff"] == 0]["block_length"]
            bl1 = sub[sub["coeff"] > 0]["block_length"]
            ax = axes[i, j] if len(sims) > 1 else axes[i]
            ax.hist(bl0, bins=30, alpha=0.5, label="coeff=0")
            ax.hist(bl1, bins=30, alpha=0.5, label="coeff>0")
            ax.set_title(f"{pop}, {sim}")
            ax.set_xlabel("# SNPs")
            ax.set_ylabel("count")
            ax.legend()
    
    fig.suptitle("# SNPs in fixed window (50000 bp)", y=0.92)

    plt.savefig("figs/snp_hist_all1.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    # stats(sys.argv[2])