# convert fasternn data to our format
import glob
import numpy as np
import matplotlib.pyplot as plt

import sys

def load(dataset="1", split="test", neut=False) -> tuple[np.ndarray, np.ndarray]:
    """Process a FASTER_NN dataset (ms) and get matrices out."""
    neut = "BASE" if neut else "TEST"
    path = f"data/FASTER_NN/D{dataset}/{split}/{neut}*.txt"
    path = glob.glob(path)

    samples = []
    snps = []

    i = 0
    hap = 0
    with open(path[0], 'r') as f:
        for lineno, line in enumerate(f):
            if line.startswith("segsites"):
                n_snps = int(line.strip().split(" ")[1])
                snps.append(n_snps)
            if line[:2] == "//":
                smp = i - 1
                i += 1
                hap = 0
            if line[:9] == "positions":
                # store distances
                dist = np.array([float(s) for s in line[11:-2].split(" ")])

            if line[0] in ["0", "1"]:
                samples[smp, hap, :] = np.array([int(s) for s in line[lower:upper]])

                hap += 1

    return snps


if __name__ == "__main__":
    split = "test"
    if len(sys.argv) > 1:
        split = sys.argv[1]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout="constrained", sharex=True)
    for i, ds in enumerate([str(x) for x in range(1, 7)]):
        neut_snps = load(ds, split, neut=True)
        sel_snps = load(ds, split, neut=False)

        ax = axs[i // 3, i % 3]
        ax.hist(neut_snps, alpha=0.5, label="Neutral")
        ax.hist(sel_snps, alpha=0.5, label="Selected")
        ax.set_title(f"Dataset {ds}")
        ax.legend()

    plt.savefig("figs/fasternn/fasternn_snps.png", dpi=300)