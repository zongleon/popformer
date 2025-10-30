"""
Some specialty dataloaders for specific datasets.
"""

import numpy as np
import sys
from datasets import Dataset, concatenate_datasets
import pandas as pd
from tqdm import tqdm
import tskit

from dataset import find_nonzero_block_cols, get_iterator, get_pos_and_dist_vec, make_features, tokenizer
from pg_gan import global_vars, util

def parse_ghist_out(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            coord = int(parts[2])
            coeff = float(parts[5])
            onset = float(parts[8])
            min_freq = float(parts[11])
            data.append((coord, coeff, onset, min_freq))
    return data

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "realsim":
        def gen():
            for pop in ["CEU", "CHB", "YRI"]:
                samples = np.load(f"../disc-interpret/dataset-{pop}/X.npy")
                labels = np.load(f"../disc-interpret/dataset-{pop}/y.npy")

                for sample, label in zip(samples, labels):
                    sample = tokenizer(sample)
                    yield {
                        "input_ids": sample[..., 0],
                        "distances": sample[0, :, 1],
                        "label": label,
                    }

        features = make_features(label_dtype="int8", label_resolution="window")
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk("dataset/ft_realsim_tkns")
    elif mode == "ancientx":
        global_vars.NUM_SNPS = 512
        global_vars.L = 50000

        def gen():
            it = get_iterator("/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
                              "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed", None)
            df = pd.read_csv(
                "ANC/Selection_Summary_Statistics_01OCT2025.tsv", comment="#", sep="\t"
            )
            df = df.set_index(["CHROM", "POS"])
            for chrom in range(1, 23):
                bound = it._chrom_bounds(chrom)
                chrom_df = df.xs(chrom, level="CHROM")
                chrom_df = chrom_df[~chrom_df.index.duplicated()]
                for i in range(0, 500_000_000, 50000):
                    pos = it.find(i, chrom)
                    if pos > bound[1]:
                        break

                    region = it.real_region(
                        neg1=False, region_len=True, start_idx=pos, return_all_pos=True
                    )
                    if region is None:
                        continue

                    region, positions, _ = region

                    xs = chrom_df["S"].reindex(positions).abs().fillna(-100)

                    sample = tokenizer(region)
                    yield {
                        "input_ids": sample[..., 0],
                        "distances": sample[0, :, 1],
                        "chrom": chrom,
                        "positions": positions,
                        "label": xs,
                    }

        features = make_features(
            label_dtype="float16", label_resolution="snp", include_snp_pos=True
        )
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk("ANC/tokenized_CHB")
    elif mode == "imputation":

        def shuffle(arr, n):
            # arr is (n_snps, n_haps)
            arr_t = arr.T  # (n_haps, n_snps)
            n_haps, n_snps = arr_t.shape
            tgt = arr_t[-n:, :]
            ref = arr_t[:-n, :]
            tgt_positions = np.random.choice(n_haps, n, replace=False)
            tgt_positions = np.sort(tgt_positions)
            new_arr = np.zeros_like(arr_t)
            new_arr[tgt_positions, :] = tgt
            remaining_positions = np.setdiff1d(np.arange(n_haps), tgt_positions)
            new_arr[remaining_positions, :] = ref
            return new_arr.T

        global_vars.NUM_SNPS = 256
        samples_list = []
        it = get_iterator("IMP/KHV.chr20.64_ref.h5", None, None)
        it2 = get_iterator("IMP/KHV.chr20.64_tgt.h5", None, None)
        n = it2.num_samples
        n_snps = it.num_snps
        n_haps_ref = it.num_samples
        n_haps = it.num_samples + it2.num_samples

        region = positions = None

        snp_ref = snp_tgt = 0
        while snp_ref < n_snps:
            cur_idx = snp_ref % global_vars.NUM_SNPS
            if cur_idx == 0:
                if snp_tgt != 0:
                    # save if not first
                    # region = shuffle(region, n)
                    dist_vec = [0] + [
                        (positions[j + 1] - positions[j]) / global_vars.L
                        for j in range(len(positions) - 1)
                    ]

                    region = util.process_gt_dist(
                        region, dist_vec, region_len=False, real=True, neg1=False
                    )

                    sample = tokenizer(region)

                    samples_list.append(
                        {
                            "input_ids": sample[..., 0],
                            "distances": sample[0, :, 1],
                            "positions": positions,
                        }
                    )

                region = np.zeros((global_vars.NUM_SNPS, n_haps))
                positions = np.zeros((global_vars.NUM_SNPS,))
            pos_ref = it.pos_all[snp_ref]
            pos_tgt = it2.pos_all[snp_tgt]

            while pos_ref < pos_tgt:
                # mask the tgt sample at this pos
                region[cur_idx, n_haps_ref:] = 4
                region[cur_idx, :n_haps_ref] = it.haps_all[snp_ref, :]
                positions[cur_idx] = it.pos_all[snp_ref]

                snp_ref += 1
                cur_idx = snp_ref % global_vars.NUM_SNPS
                pos_ref = it.pos_all[snp_ref]

            region[cur_idx, n_haps_ref:] = it2.haps_all[snp_tgt, :]
            region[cur_idx, :n_haps_ref] = it.haps_all[snp_ref, :]
            positions[cur_idx] = it.pos_all[snp_ref]

            snp_ref += 1
            snp_tgt += 1

        # Handle the last region if it wasn't saved
        if region is not None and positions is not None:
            region = shuffle(region, n)
            dist_vec = [0] + [
                (positions[j + 1] - positions[j]) / global_vars.L
                for j in range(len(positions) - 1)
            ]
            region = util.process_gt_dist(
                region, dist_vec, region_len=False, real=True, neg1=False
            )
            sample = tokenizer(region)
            samples_list.append(
                {
                    "input_ids": sample[..., 0],
                    "distances": sample[0, :, 1],
                    "positions": positions,
                }
            )

        features = make_features(include_snp_pos=True)
        # Save tokenized data
        dataset = Dataset.from_list(samples_list, features=features)
        dataset.save_to_disk(f"IMP/KHV.chr20.64.{global_vars.NUM_SNPS}")

    elif mode == "fasternn":
        global_vars.NUM_SNPS = 512
        split = "test"
        def gen():
            samples = np.load(f"FASTER_NN/fasternn_{split}_regions_50000.npy")
            distances = np.load(f"FASTER_NN/fasternn_{split}_distances_50000.npy")
            labels = pd.read_csv(f"FASTER_NN/fasternn_{split}_meta.csv")["label"].values
            for sample, distance in zip(samples, distances):
                region = np.dstack(
                    [sample, distance[None, :].repeat(sample.shape[0], axis=0)]
                )
                region, distances = tokenizer(region)
                yield {"input_ids": region, "distances": distances, "labels": labels}

        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=make_features(label_dtype="int8", label_resolution="window"))
        dataset.save_to_disk(f"FASTER_NN/tokenized_{split}_50000")
    elif mode == "runsel":
        which = sys.argv[2]
        allsamples: np.ndarray = np.load(
            f"1000g/{which}/matrices.npy", mmap_mode="r"
        )
        alldistances: np.ndarray = np.load(
            f"1000g/{which}/distances.npy", mmap_mode="r"
        )
        df = pd.read_csv(f"1000g/{which}/metadata.csv", memory_map=True)
        global_vars.NUM_SNPS = 512

        def gen():
            # sel = df["coeff"].to_numpy(dtype=np.float16)
            sel = df["coeff"].to_numpy()
            sel = (sel > 0).astype(int)

            for i, (sample, dist, s) in enumerate(zip(allsamples, alldistances, sel)):
                # only nonzero
                first, last = find_nonzero_block_cols(sample)
                sample = sample[:, first:last]
                dist = dist[first:last]

                sample = np.dstack(
                    [sample, dist[None, :].repeat(sample.shape[0], axis=0)]
                )
                region, distances = tokenizer(sample)

                yield {
                    "input_ids": region,
                    "distances": distances,
                    "label": s,
                }

        features = make_features(label_dtype="float16", label_resolution="window")

        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk(f"dataset/selbin_{which}")
    elif mode == "runsel_bigregion":
        rng = np.random.default_rng()
        def gen(fpath: str, mpath: str):
            # process tree
            ts = tskit.load(fpath)
            gt_matrix = ts.genotype_matrix()
            num_snps = gt_matrix.shape[0]
            positions, dist_vec = get_pos_and_dist_vec(ts, num_snps)

            meta = parse_ghist_out(mpath)

            gt_matrix = gt_matrix.T

            cum_pos = np.cumsum(dist_vec)

            last_pos = int(cum_pos[-1])
            start_bp = 0
            for _ in range(5000):
                start_bp = np.random.randint(0, last_pos - 64)
                start_idx = int(np.searchsorted(cum_pos, start_bp, side="left"))
                length = rng.integers(low=16, high=64)
                end_idx = min(start_idx + length, gt_matrix.shape[1])
                end_bp = cum_pos[end_idx - 1]
                # tqdm.write(f"Sampling window {start_bp}-{end_bp} (idx {start_idx}-{end_idx})")

                m = gt_matrix[:, start_idx:end_idx]
                d = dist_vec[start_idx:end_idx].copy()
                p = positions[start_idx:end_idx]
                d[0] = 0

                dist = d[None, :].repeat(m.shape[0], axis=0)
                region = np.dstack([m, dist])
                region, distances = tokenizer(region)

                # find label
                label = 0.0
                for coord, coeff, onset, min_freq in meta:
                    if start_bp <= coord - 10000000 <= end_bp:
                        label = 1.0
                        tqdm.write(f"Found sel at coord {coord} with coeff {coeff}")
                        break

                yield {
                    "input_ids": region,
                    "distances": distances,
                    "chrom": 0,
                    "positions": p,
                    "label": label,
                }


        features = make_features(label_dtype="float16", label_resolution="window", include_snp_pos=True)
        dataset = Dataset.from_generator(lambda: gen("1000g/regiontest/ghist_const4.trees",
                                                     "1000g/regiontest/ghist_const4.out"),
                                         features=features)
        dataset2 = Dataset.from_generator(lambda: gen("1000g/regiontest/ghist_const6.trees",
                                                      "1000g/regiontest/ghist_const6.out"),
                                          features=features)
        dataset = concatenate_datasets([dataset, dataset2])
        dataset.save_to_disk("dataset/selbin_bigregion")