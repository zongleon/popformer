import numpy as np
import sys
import os
import pandas as pd
import tskit
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(".")
from popformer.dataset import find_nonzero_block_cols

FILE = "{pop}_{typ}{seed}.trees"
MAX_HAPS = 256
MAX_SNPS = 4096


def main(d, out, pop="human"):
    # paths
    d = os.path.dirname(d)
    metadata = os.path.join(d, "sweep_metadata.csv")
    os.makedirs(out, exist_ok=True)

    # input metadat
    df = pd.read_csv(metadata)
    df.columns = [col.strip() for col in df.columns.tolist()]
    # neutral cols
    #   seed, pop, mut, reco, dfe
    # selection columns
    #   coordinate, coeff, onset_time, goal_freq
    total = df.shape[0]

    # store results
    matrices = np.full((total, MAX_HAPS, MAX_SNPS), 5, dtype=np.int8)
    distances = np.zeros((total, MAX_SNPS), dtype=np.int32)
    ns = []
    for idx, row in tqdm(df.iterrows(), total=total):
        sel = row["coeff"] > 0
        filename = os.path.join(
            d, FILE.format(pop=pop, typ="sweep" if sel else "neutral", seed=row["seed"])
        )

        # process tree
        ts = tskit.load(filename)
        gt_matrix = ts.genotype_matrix()
        # pg-pfn @sara
        is_biallelic = [
            sum(gt_matrix[i]) == list(gt_matrix[i]).count(1)
            for i in range(len(gt_matrix))
        ]
        gt_matrix = gt_matrix[is_biallelic]
        tqdm.write(f"Processing {filename}: {gt_matrix.shape[0]} SNPs")
        num_snps = min(gt_matrix.shape[0], MAX_SNPS)
        num_haps = min(gt_matrix.shape[1], MAX_HAPS)
        dist_vec = get_dist_vec(ts, mask=is_biallelic)
        gt_matrix = gt_matrix.T

        matrices[idx, :num_haps, :num_snps] = gt_matrix[:num_haps, :num_snps]
        distances[idx, :num_snps] = dist_vec[:num_snps]

        ns.append(num_snps)

    print("\n\nTrees processed.")
    print(f"Haps: {num_haps}")
    print(f"SNPs: avg {np.mean(ns)} ({min(ns)} - {max(ns)})")

    np.save(os.path.join(out, "matrices.npy"), matrices)
    np.save(os.path.join(out, "distances.npy"), distances)
    df.to_csv(os.path.join(out, "metadata.csv"), index=False)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
    sub = df
    snps = np.array(ns)
    bl0 = snps[sub["coeff"] == 0]
    bl1 = snps[sub["coeff"] > 0]
    ax = axes
    ax.hist(bl0, bins=30, alpha=0.5, label="coeff=0")
    ax.hist(bl1, bins=30, alpha=0.5, label="coeff>0")
    ax.set_xlabel("# SNPs")
    ax.set_ylabel("count")
    ax.legend()

    plt.savefig(f"figs/snpdists/{pop}.png", dpi=300, bbox_inches="tight")


def get_dist_vec(ts, mask=None):
    positions = [round(variant.site.position) for variant in ts.variants()]
    positions = [pos for i, pos in enumerate(positions) if mask is None or mask[i]]
    snps_total = len(positions)

    dist_vec = [0] + [(positions[j + 1] - positions[j]) for j in range(snps_total - 1)]
    return np.array(dist_vec)


if __name__ == "__main__":
    # take input dir and output dir as args
    main(sys.argv[1], sys.argv[2], pop=sys.argv[3] if len(sys.argv) > 3 else "human")
