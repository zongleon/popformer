import os
import numpy as np
import pandas as pd
from popformer.dataset import find_nonzero_block_cols

INPUT = "data/matrices/pan_4/"
INPUT_MATS = os.path.join(INPUT, "matrices.npy")
INPUT_DISTANCES = os.path.join(INPUT, "distances.npy")
INPUT_METADATA = os.path.join(INPUT, "metadata.csv")

OUTPUT = "../diploSHIC/data/"

matrices = np.load(INPUT_MATS)
distances = np.load(INPUT_DISTANCES)
metadata = pd.read_csv(INPUT_METADATA)

print(f"Loaded {matrices.shape[0]} matrices of shape {matrices.shape[1:]}")
print(f"Loaded {distances.shape[0]} distance arrays of shape {distances.shape[1:]}")
print(f"Loaded metadata with {metadata.shape[0]} entries and columns: {metadata.columns.tolist()}")

for simtype in ["neutral", "hard"]:
    if simtype == "neutral":
        indices = metadata.index[metadata["coeff"] == 0].tolist()
    else:
        indices = metadata.index[metadata["coeff"] > 0].tolist()
    sim_matrices = matrices[indices]
    sim_distances = distances[indices]
    
    f = open(os.path.join(OUTPUT, f"{simtype}_2.msOut.txt"), "w")
    
    # we're writing ms style output
    # first line is 'program numSamples numSims'
    # index of the first all-zero row in the first matrix

    first_mat = sim_matrices[0]
    zero_rows = np.all(first_mat == 0, axis=1)
    zero_idx = np.where(zero_rows)[0][0]
    f.write(f"popformer-sims-ms {zero_idx} {sim_matrices.shape[0]}\n")
    f.write("\n")

    for i in range(sim_matrices.shape[0]):
        mat = sim_matrices[i]
        dist = sim_distances[i]
        # find_nonzero_block_cols returns (first_nonzero_col, last_nonzero_col)
        start_col, end_col = find_nonzero_block_cols(mat)
        nonzero_cols = list(range(start_col, end_col + 1))
        mat = mat[:, nonzero_cols]
        
        # write segsites line and separator
        f.write("//\n")
        f.write(f"segsites: {len(nonzero_cols)}\n")
        
        # write positions line
        positions = np.cumsum(dist)
        positions = positions[nonzero_cols] / positions[end_col]
        positions[-1] = positions[-1] - 1e-6
        positions_str = " ".join([f"{pos:.6f}" for pos in positions])
        f.write(f"positions: {positions_str}\n")
        
        # write haplotype lines
        for row in mat:
            if (row == 0).all():
                continue
            haplotype = "".join([str(int(x)) for x in row])
            f.write(f"{haplotype}\n")
        
        f.write("\n")

    f.close()
    print(f"Wrote {simtype} simulations to {os.path.join(OUTPUT, f'{simtype}_2.msOut.txt')}")

