import numpy as np
import sys
import os
import pandas as pd
import tskit
from tqdm import tqdm

sys.path.append(".")
from dataset import find_nonzero_block_cols
from pg_gan import global_vars

MIDDLE_ONLY = True

FILE = "{pop}_{typ}{seed}.trees"
MAX_HAPS = 256
MAX_SNPS = 64 if MIDDLE_ONLY else 512

def main(dir, out):
    # paths
    dir = os.path.dirname(dir)
    pop = dir.split("/")[-1]
    metadata = os.path.join(dir, "sweep_metadata.csv")
    os.makedirs(out, exist_ok=True)

    # input metadat
    df = pd.read_csv(metadata)
    df.columns = [col.strip() for col in df.columns.tolist()]
    n_seeds = df.shape[0]
    total = n_seeds * 2
    print(f"\nLoaded metadata: {n_seeds} seeds ({total} total)")
    print(f"\tColumns: {df.columns.tolist()}")

    # update metadata
    neutrals = [(seed, -1, 0, 0, 0) for seed in range(n_seeds)]
    neutral_df = pd.DataFrame(neutrals, columns=df.columns)
    df = pd.concat([df, neutral_df])
    df["sim"] = "Sept25"
    df["pop"] = pop
    
    # store results
    matrices = np.zeros((total, MAX_HAPS, MAX_SNPS))
    distances = np.zeros((total, MAX_SNPS))
    ns = []

    i = 0
    pbar = tqdm(total=total)

    for subdir in ["sweep", "neutral"]:
        for seed in range(n_seeds):
            pbar.update(1)
            filename = os.path.join(dir, subdir, FILE.format(pop=pop, 
                                                             typ=subdir,
                                                             seed=seed))
            
            
            # process tree        
            ts = tskit.load(filename)
            gt_matrix = ts.genotype_matrix()
            num_snps = gt_matrix.shape[0]
            num_haps = gt_matrix.shape[1]
            dist_vec = get_dist_vec(ts, num_snps)

            gt_matrix = gt_matrix.T

            if MIDDLE_ONLY:
                first, last = find_nonzero_block_cols(gt_matrix)
                
                # take middle SNPs
                mid = (first + last) // 2
                half = 32 # middle 64 SNPs
                first = max(0, mid - half)
                last = min(num_snps, mid + half)

                # store
                matrices[i, :num_haps, :(last-first)] = gt_matrix[:, first:last]
                distances[i, :(last-first)] = dist_vec[first:last]
            else:
                matrices[i, :num_haps, :num_snps] = gt_matrix
                distances[i, :num_snps] = dist_vec

            ns.append(num_snps)

            i += 1
            
    pbar.close()

    print("\n\nTrees processed.")
    print(f"Haps: {num_haps}")
    print(f"SNPs: avg {np.mean(ns)} ({min(ns)} - {max(ns)})")

    np.save(os.path.join(out, "matrices.npy"), matrices)
    np.save(os.path.join(out, "distances.npy"), distances)
    df.to_csv(os.path.join(out, "metadata.csv"), index=False)


def get_dist_vec(ts, snps_total):
    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    
    dist_vec = [0] + [(positions[j+1] - positions[j])/ \
              global_vars.L for j in range(snps_total-1)]
    return np.array(dist_vec)


if __name__ == "__main__":
    # take input dir and output dir as args
    main(sys.argv[1], sys.argv[2])
