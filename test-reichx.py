import sys
import numpy as np
import torch
from scipy.stats import spearmanr, permutation_test
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def get_windows(preds, mask=None):
    chroms = preds["chrom"]
    starts = preds["start_pos"] 
    ends = preds["end_pos"]

    if mask is not None:
        chroms = chroms[mask]
        starts = starts[mask]
        ends = ends[mask]

    return list(zip(chroms, starts, ends))

def get_windowed_reich(windows):
    df = pd.read_csv(
        "ANC/Selection_Summary_Statistics_01OCT2025.tsv", comment="#", sep="\t"
    )
    df = df.set_index(["CHROM", "POS"])

    # we'll compute the max of column "X" for each window
    reichx = []
    for chrom, start, end in tqdm(windows):
        window_df = df.loc[(chrom, slice(start, end)), :]
        if window_df.empty:
            reichx.append(0)
            continue
        reich_stat = window_df["X"].mean()
        reichx.append(reich_stat)

    assert len(reichx) == len(windows)
    reichx = np.array(reichx)
    return reichx

def get_windowed_grossman(windows):
    df = pd.read_csv("SEL/sel.csv")
    df = df[df["Population"] == "CEU"]
    # convert "chr1", etc
    df["Chromosome"] = df["Chromosome"].apply(lambda x: int(x.replace("chr", "")))

    df = df.set_index(["Chromosome", "Start"])
    df = df.sort_index()

    grossman = []
    
    for chrom, start, end in tqdm(windows):
        try:
            window_df = df.loc[
                (chrom,
                slice(start, end)), :
            ]
        except KeyError:
            window_df = pd.DataFrame()
        
        if window_df.empty:
            grossman.append(0)
        else:
            grossman.append(1)

    grossman = np.array(grossman)
    return grossman

def region_shift_test(X, region_mask, M=10000, seed=None, roll=False):
    rng = np.random.default_rng(seed)
    obs = X[region_mask].mean()
    n = len(X)
    null = np.empty(M)
    for b in range(M):
        if roll:
            shift = rng.integers(0, n)
            null[b] = X[np.roll(region_mask, shift)].mean()
        else:
            null_mask = rng.permutation(region_mask)
            null[b] = X[null_mask].mean()

    p_emp = (1 + np.sum(null >= obs)) / (1 + M)    # one-sided
    fe = obs / null.mean()
    ci_lo, ci_hi = np.percentile(null, [2.5, 97.5])
    # optionally convert to FE CI by dividing obs by null percentiles (note asymmetry)
    fe_ci = (obs / ci_hi, obs / ci_lo)

    print(f"Observed mean: {obs:.4f}")
    print(f"Empirical p-value: {p_emp:.4g}")
    print(f"Fold enrichment: {fe:.4f}")
    print(f"95% CI for null mean: ({ci_lo:.4f}, {ci_hi:.4f})")
    print(f"95% CI for fold enrichment: ({fe_ci[0]:.4f}, {fe_ci[1]:.4f})")
    

if __name__ == "__main__":
    preds = sys.argv[1]
    preds = np.load(preds)

    # mask = preds["chrom"]
    mask = np.ones_like(preds["chrom"], dtype=bool)
    windows = get_windows(preds, mask)
    reichx = get_windowed_reich(windows)
    grossman = get_windowed_grossman(windows)

    print(f"Got {len(windows)} windows.")

    def to_probs(x):
        x = torch.tensor(x)
        return torch.softmax(x, dim=1).numpy()[:, 1]

    preds = to_probs(preds["preds"])
    reich_x_sig_mask = reichx >= 4.5
    grossman_sig_mask = grossman == 1
    sig_mask = grossman_sig_mask

    preds_sig = preds[mask][sig_mask]
    preds_nonsig = preds[mask][~sig_mask][: len(preds_sig)]

    # print(preds_sig.shape, preds_nonsig.shape)

    region_shift_test(preds, grossman_sig_mask, M=10000, roll=True)
    region_shift_test(preds, reich_x_sig_mask)

    # res = spearmanr(preds, reichx)

    # print(f"Spearman r: {res.correlation:.4f}, p-value: {res.pvalue:.4g}")