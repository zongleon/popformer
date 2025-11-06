import sys
import pandas as pd

WINDOW = 0

path = sys.argv[1]
outpath = sys.argv[2]

data = []
with open(path, "r") as f:
    for line in f:
        region = line.strip().split(" ")
        data.append((int(region[2]) - WINDOW, int(region[2]) + WINDOW))

df = pd.DataFrame(data, columns=["start", "end"])
df.to_csv(outpath, index=False)