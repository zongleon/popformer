# convert fasternn data to our format
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, ".")
from pg_gan import util

def load(dataset="1", split="test", neut=False) -> tuple[np.ndarray, np.ndarray]:
    """Process a FASTER_NN dataset (ms) and get matrices out."""
    neut = "BASE" if neut else "TEST"
    path = f"FASTER_NN/D{dataset}/{split}/{neut}*.txt"
    path = glob.glob(path)

    n_haps = 128
    n_snps = 512
    total = 1000
    pbar = tqdm(total=total)

    samples = np.zeros((total, n_haps, n_snps), dtype=np.int8)
    distances = np.zeros((total, n_snps), dtype=np.float32)

    use_site = False
    with open(path[0], 'r') as f:
        for lineno, line in enumerate(f):
            if lineno < 2:
                pbar.write(f"{line.strip()}")
                if "-s" in line:
                    use_site = True
                continue
            if line[:2] == "//":
                pbar.update(1)
                smp = pbar.n - 1
                hap = 0
            if line[:9] == "positions":
                # store distances
                dist = np.array([float(s) for s in line[11:-2].split(" ")])

                lower = max(len(dist) // 2 - 256, 0)
                upper = min(len(dist) // 2 + 256, len(dist))

                # pbar.write(f"[{lower}, {upper}] ({upper - lower})")

                scale_factor = 50000
                distances[smp, :(upper - lower)] *= scale_factor
                distances[smp, :(upper-lower)] = (dist[lower:upper] - dist[lower])

            if line[0] in ["0", "1"]:
                samples[smp, hap, :(upper-lower)] = np.array([int(s) for s in line[lower:upper]])

                hap += 1

    pbar.close()

    return samples, distances


if __name__ == "__main__":
    split = "test"
    if len(sys.argv) > 1:
        split = sys.argv[1]
    metas = []
    samples = []
    distances = []

    for ds in [str(x) for x in range(1, 7)]:
        s, d = load(ds, split, neut=True)
        
        metas.extend([(f"D{ds}", 0)] * s.shape[0])
        samples.append(s)
        distances.append(d)

        s, d = load(ds, split, neut=False)
        
        metas.extend([(f"D{ds}", 1)] * s.shape[0])
        samples.append(s)
        distances.append(d)

    samples = np.concatenate(samples, axis=0)
    distances = np.concatenate(distances, axis=0)
    meta = pd.DataFrame(metas, columns=["dataset", "label"])

    # maj-min
    for i in range(samples.shape[0]):

        samples[i] = util.major_minor(samples[i], False)

    print(samples.shape, distances.shape)

    np.save(f"FASTER_NN/fasternn_{split}_regions_512snps.npy", samples)
    np.save(f"FASTER_NN/fasternn_{split}_distances_512snps.npy", distances)
    meta.to_csv(f"FASTER_NN/fasternn_{split}_meta.csv", index=False)
