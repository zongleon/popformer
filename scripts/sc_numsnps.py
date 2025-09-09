# sanity check: num snps in selected vs unselected

import matplotlib.pyplot as plt
import numpy as np

def find_nonzero_block_cols(sample: np.ndarray) -> tuple[int, int]:
    """
    Find the first and last col indices in a 2D array that are not all zeros.
    Returns (first_idx, last_idx), inclusive.
    If all cols are zero, returns (None, None).
    """
    # sample: shape (n_rows, n_cols)
    nonzero_mask = ~(np.all(sample == 0, axis=0))
    nonzero_indices = np.where(nonzero_mask)[0]
    if nonzero_indices.size == 0:
        return (None, None)
    return (nonzero_indices[0], nonzero_indices[-1])

nsnps = []

for t, sel, sel_bin in zip(
        ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
        [0, 0.1, 0.01, 0.05, 0.025],
        [0, 1, 1, 1, 1]
    ):
    for pop_idx, pop in enumerate(["CEU", "CHB", "YRI"]):
        samples: np.ndarray = np.load(f"1000g/{pop}/matrices_regions_{pop}_{t}.npy")[:2400]
        distances: np.ndarray = np.load(f"1000g/{pop}/distances_regions_{pop}_{t}.npy")[:2400]

        for sample, dist in zip(samples, distances):
            sample = sample.T
            first, last = find_nonzero_block_cols(sample)
            
            nsnps.append((last - first, sel_bin, pop_idx))

nsnps_arr = np.array(nsnps)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
bins = np.linspace(0, 512, 32)
pops = ["CEU", "CHB", "YRI"]

for ax, sel_bin, title in zip(axes, [0, 1], ['neutral', 'selection']):
    for pop_idx, pop in enumerate(pops):
        data = nsnps_arr[(nsnps_arr[:, 1] == sel_bin) & (nsnps_arr[:, 2] == pop_idx)][:, 0]
        ax.hist(data, bins=bins, alpha=0.7, label=pop)
    ax.set_xlabel('Number of SNPs')
    ax.set_title(title)
axes[0].set_ylabel('Count')
axes[1].legend()
fig.suptitle('Histogram of SNP counts by population')
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figs/snp_hist_sel_by_pop.png", dpi=300, bbox_inches="tight")