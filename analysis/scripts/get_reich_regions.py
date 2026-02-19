import sys
import pandas as pd
from tqdm.rich import tqdm


PATH = sys.argv[1]

df = pd.read_csv(PATH, sep="\t", comment="#")

print(df.head())

df = df[["CHROM", "POS", "X"]]

# for 50kb windows, add windows that have mean X < 2
tups = []
for chrom in df["CHROM"].unique():
    chrom_df = df[df["CHROM"] == chrom]
    for start in tqdm(range(0, chrom_df["POS"].max(), 50000)):
        end = start + 50000
        window_df = chrom_df[(chrom_df["POS"] >= start) & (chrom_df["POS"] < end)]
        x_agg = window_df["X"].abs().max()
        if x_agg < 2:
            tups.append((chrom, start, end, x_agg))

# let's just take chr1
df = pd.DataFrame(tups, columns=["Chromosome", "Start", "End", "X_agg"])
df["Population"] = "CEU"

df.to_csv("data/SEL/reichsel_negs.csv", index=False)
