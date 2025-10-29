import numpy as np
import sys
import os
import pandas as pd
import tskit
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(".")
from dataset import find_nonzero_block_cols
from pg_gan import global_vars

MAX = 5000

FILE = "{pop}_{typ}{seed}.trees"
MAX_HAPS = 256
MAX_SNPS = 512

def main(d, out):
    # paths
    d = os.path.dirname(d)
    metadata = os.path.join(d, "sweep_metadata.csv")
    metadata2 = os.path.join(d, "neutral_metadata.csv")
    os.makedirs(out, exist_ok=True)

    # input metadat
    df = pd.read_csv(metadata)
    df.columns = [col.strip() for col in df.columns.tolist()]
    n_seeds = min(MAX, df.shape[0])
    df = df.iloc[:n_seeds]
    total = n_seeds * 2
    print(f"\nLoaded metadata: {n_seeds} seeds ({total} total)")
    print(f"\tColumns: {df.columns.tolist()}")

    # update metadata
    neutrals = [(-1, 0, 0, 0) for _ in range(n_seeds)]
    neutral_df = pd.read_csv(metadata2).iloc[:n_seeds]
    neutral_df = pd.concat([neutral_df, pd.DataFrame(neutrals, columns=df.columns[-4:])], axis=1)
    df = pd.concat([df, neutral_df])
    df["sim"] = "Sept25"
    df["pop"] = "pan_2"
    
    # store results
    matrices = np.zeros((total, MAX_HAPS, MAX_SNPS))
    distances = np.zeros((total, MAX_SNPS))
    ns = []

    i = 0
    pbar = tqdm(total=total)

    for subdir in ["sweep", "neutral"]:
        for seed in range(n_seeds):
            pbar.update(1)
            filename = os.path.join(d, subdir, FILE.format(pop="human", 
                                                             typ=subdir,
                                                             seed=seed))
            
            
            # process tree        
            ts = tskit.load(filename)
            gt_matrix = ts.genotype_matrix()
            num_snps = min(gt_matrix.shape[0], MAX_SNPS)
            num_haps = gt_matrix.shape[1]
            dist_vec = get_dist_vec(ts)[:num_snps]

            gt_matrix = gt_matrix.T

            matrices[i, :num_haps, :num_snps] = gt_matrix[:, :num_snps]
            distances[i, :num_snps] = dist_vec[:num_snps]

            ns.append(num_snps)

            i += 1
            
    pbar.close()

    print("\n\nTrees processed.")
    print(f"Haps: {num_haps}")
    print(f"SNPs: avg {np.mean(ns)} ({min(ns)} - {max(ns)})")

    np.save(os.path.join(out, "matrices.npy"), matrices)
    np.save(os.path.join(out, "distances.npy"), distances)
    df.to_csv(os.path.join(out, "metadata.csv"), index=False)


def get_dist_vec(ts):
    positions = [round(variant.site.position) for variant in ts.variants()]
    snps_total = len(positions)
    
    dist_vec = [0] + [(positions[j+1] - positions[j])/ \
              global_vars.L for j in range(snps_total-1)]
    return np.array(dist_vec)


def stats(path):
    matrices = np.load(os.path.join(path, "matrices.npy"))
    df = pd.read_csv(os.path.join(path, "metadata.csv"))

    # Compute nonzero block cols for each matrix
    block_lengths = []
    for mat in matrices:
        first, last = find_nonzero_block_cols(mat)
        block_lengths.append(last - first)
    df["block_length"] = block_lengths

    lowmuts = df["low_mut"].unique() if "low_mut" in df.columns else ["low_mut"]

    fig, axes = plt.subplots(len(lowmuts), 1, figsize=(8, 12), sharex=True)
    for i, lowmut in enumerate(lowmuts):
        sub = df[df["low_mut"] == lowmut]
        bl0 = sub[sub["coeff"] == 0]["block_length"]
        bl1 = sub[sub["coeff"] > 0]["block_length"]
        ax = axes[i]
        ax.hist(bl0, bins=30, alpha=0.5, label="coeff=0")
        ax.hist(bl1, bins=30, alpha=0.5, label="coeff>0")
        ax.set_title(f"low_mut={lowmut}")
        ax.set_xlabel("# SNPs")
        ax.set_ylabel("count")
        ax.legend()
    
    fig.suptitle("# SNPs in fixed window (50000 bp)", y=0.92)

    plt.savefig("figs/snp_hist_pan.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # take input dir and output dir as args
    main(sys.argv[1], sys.argv[2])
    # stats(sys.argv[2])
