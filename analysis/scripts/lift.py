import pandas as pd

INPUT_TSV = "data/recomb/decode_recomb_hg18.tsv"
OUTPUT_BED = "data/recomb/decode_recomb_hg18_hotspots.bed"

df = pd.read_csv(INPUT_TSV, sep="\t")

# BED: chr, start (0-based), end, name
bed = pd.DataFrame(
    {
        "chr": df["chr"],
        "start": df["pos"] - 5000,
        "end": df["pos"] + 5000,
        "name": df.index.astype(str),
        "score": df["stdrate"],
    }
)

# keep hotspots with score > 10
bed = bed[bed["score"] > 10]

bed.to_csv(OUTPUT_BED, sep="\t", header=False, index=False)
