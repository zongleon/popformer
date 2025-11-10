import pandas as pd

df = pd.read_excel("SEL/selected_regions.xlsx")
df = df[["HG19 co-ordinates", "Population"]]

df["Chromosome"] = df["HG19 co-ordinates"].apply(lambda x: x.split(":")[0])
df["Start"] = df["HG19 co-ordinates"].apply(lambda x: x.split(":")[1].split("-")[0]).astype(int)
df["End"] = df["HG19 co-ordinates"].apply(lambda x: x.split(":")[1].split("-")[1]).astype(int)


# let's just take chr1
# df = df[df["Chromosome"] == "chr1"]
df = df.drop(columns="HG19 co-ordinates")
df["Population"] = df["Population"].str.replace("CHBJPT", "CHB")

bed_df = df[["Chromosome", "Start", "End"]].copy()
bed_df["Start"] = bed_df["Start"] - 1000000
bed_df["End"] = bed_df["End"] + 1000000
bed_df.to_csv("SEL/bed.bed", index=False, sep="\t", header=False)

df.to_csv("SEL/sel.csv", index=False)
