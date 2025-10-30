import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

sys.path.append(".")
from dataset import find_nonzero_block_cols

FILE = "{pop}_{typ}{seed}.trees"
MAX_HAPS = 256
MAX_SNPS = 512

def main(out):
    cols = ["seed", "coordinmate", "coeff"]
    df = pd.DataFrame(columns=cols)
    all_mats = []
    all_dists = []

    for t, sel in zip(
        ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
        [0, 0.1, 0.01, 0.05, 0.025],
    ):
        for pop in ["CEU", "CHB", "YRI"]:
            print(f"Processing {pop} {t}")
            lm = np.load(f"1000g/{pop}/matrices_regions_{pop}_{t}.npy")
            ld = np.load(f"1000g/{pop}/distances_regions_{pop}_{t}.npy")
            
            n = lm.shape[0]
            n_snps = min(lm.shape[1], MAX_SNPS)
            n_haps = lm.shape[2]

            # store results
            matrices = np.zeros((n, MAX_HAPS, MAX_SNPS), dtype=np.float16)
            distances = np.zeros((n, MAX_SNPS), dtype=np.float16)
            
            for idx, (mat, dist) in enumerate(zip(lm, ld)):
                mat = mat.T

                matrices[idx, :n_haps, :n_snps] = mat[:, :n_snps]
                distances[idx, :n_snps] = dist[None, :n_snps]

            # metadata
            data = [(seed, -1 if "neutral" in t else 25000,
                     sel) for seed in range(n)]
            
            # combine!
            df = pd.concat([df, pd.DataFrame(data, columns=cols)])
            all_mats.append(matrices)
            all_dists.append(distances)

    np.save(os.path.join(out, "matrices.npy"), np.concatenate(all_mats))
    np.save(os.path.join(out, "distances.npy"), np.concatenate(all_dists))
    df.to_csv(os.path.join(out, "metadata.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1])