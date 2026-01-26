import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from popformer.real_data_random import RealDataRandomIterator, Region

# use pg gan iterator to get region
it = RealDataRandomIterator(
    filename="/bigdata/smathieson/1000g-share/VCF_Aug24/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
    bed_file="/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed",
)
starts, ends, chromosomes, fractions = [], [], [], []
chroms = list(range(1, 23))
for chrom in chroms:
    bound = it._chrom_bounds(chrom)
    print(f"{chrom} | {bound[0]} - {bound[1]}")
    pos, i = 0, 0
    while pos < bound[1] - 1:
        if chrom == chroms[0] and i == 0:
            pbar = tqdm()
        pos = it.find(i, chrom)
        reg = it.real_region(512, True, 50000, pos, return_pos=True)

        i = i + 50000

        if reg == "end_chrom":
            break
        if reg is None:
            continue

        # for every 50kb, store the callable fraction
        region = Region(chrom, i, i + 50000)
        fraction = region.inside_mask(it.mask_dict, return_fraction=True)

        starts.append(region.start_pos)
        ends.append(region.end_pos)
        chromosomes.append(int(region.chrom))
        fractions.append(fraction)

        tqdm.write(
            f"Chrom {chrom} | pos: {pos} ({bound[0]} - {bound[1]}) | i: {i} | frac: {fraction:.4f}"
        )

        pbar.update(1)

starts = np.array(starts)
ends = np.array(ends)
chromosomes = np.array(chromosomes)
fractions = np.array(fractions)

# plot as manhattan plot ala sweep.py
chroms = sorted(np.unique(chromosomes))

# Chromosome lengths and cumulative offsets
lengths = {c: ends[chromosomes == c].max() for c in chroms}
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

fig, ax = plt.subplots(1, 1, figsize=(20, 3), layout="constrained")

for c in chroms:
    mask = chromosomes == c
    if not np.any(mask):
        continue
    x = starts[mask] + offsets[c]
    y = fractions[mask]
    ax.scatter(
        x,
        y,
        s=5,
        color=colors((c - 1) % 2),
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )

# ax.set_ylim(0, 1)
ax.set_ylabel("fraction callable")
ax.set_title("50kb windows")
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

ax.legend(loc="upper right")
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel("Chromosome")

plt.savefig("figs/fraction_callable_manhattan.png", dpi=300)
plt.close(fig)

np.savez(
    "preds/genome_CAL_sel_popf-lp-pan_region_plot_data.npz",
    start_pos=starts,
    end_pos=ends,
    chrom=chromosomes,
    preds=fractions,
)

print(f"preds shape: {fractions.shape}")
