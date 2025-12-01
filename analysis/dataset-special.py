"""
Some specialty dataloaders for specific datasets.
"""

import numpy as np
import sys
from datasets import Dataset, concatenate_datasets
import pandas as pd
from tqdm import tqdm
import tskit

from popformer.dataset import (
    Tokenizer,
    get_pos_and_dist_vec,
    make_features,
)


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

def gen_sel(samps, dists, meta, meta_dict = {}, times=1, force_s=False, start_at=None):
    sel = meta["coeff"].to_numpy(dtype=np.float16)
    pos = meta["coordinate"].to_numpy(dtype=np.int32)
    
    for _ in range(times):
        for i, (sample, dist, s, p) in enumerate(zip(samps, dists, sel, pos)):
            # only nonzero
            pad_snp = (sample == 5).all(axis=0)
            # find first row that's all 5s
            pad_hap = (sample == 5).all(axis=1)
            sample = sample[~pad_hap][:, ~pad_snp]
            dist = dist[~pad_snp]

            positions = np.cumsum(dist)
            # tqdm.write(f"{~pad_snp.sum()}: {positions}")

            # tqdm.write(f"{sample.shape}, dist {dist.shape}, pos {positions.shape}")
            found = False
            while not found:
                length = sample.shape[1]
                start_idx = np.random.randint(0, max(1, length - 128)) if start_at is None else start_at
                # get 50k region
                end_idx = (
                    np.searchsorted(positions, positions[start_idx] + 50000)
                )
                # end_idx = min(start_idx + region_len, length)
                found_sample = sample[:, start_idx:end_idx]
                found_dist = dist[start_idx:end_idx]

                # if coordinate is not in the middle of the window,
                # sel should be 0
                if s == 0 or positions[end_idx - 1] < p or positions[start_idx] > p:
                    found = True
                if positions[start_idx] <= p <= positions[end_idx - 1]:
                    if force_s is None or force_s:
                        found = True
                else:
                    if force_s is None or not force_s:
                        s = 0
                        found = True
                if found:
                    tqdm.write(
                        f"Sample {i}: pos {positions[start_idx]}-{positions[end_idx - 1]}, s={s}, p={p}"
                        f"\n{start_idx}-{end_idx}, length {length}"
                        # f"shapes {found_sample.shape}, {found_dist.shape}"
                    )

            assert found_sample.shape[1] > 0, (
                f"Empty region for sample {i}, start {start_idx}, end {end_idx}, shape {sample.shape}"
            )

            sample = np.dstack(
                [
                    found_sample,
                    found_dist[None, :].repeat(found_sample.shape[0], axis=0),
                ]
            )
            region, distances = tokenizer(sample)
            # tqdm.write(f"label {1 if s > 0 else 0}, s={s}")

            result =  {
                "input_ids": region,
                "distances": distances,
                "label": 1 if s > 0 else 0,
                "s": s,
            }

            for meta_key, dtype in meta_dict.items():
                result[meta_key] = np.array(meta.iloc[i][meta_key])
            
            yield result

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
        raise SystemExit
        # global_vars.NUM_SNPS = 512
        # global_vars.L = 50000

        # def gen():
        #     it = get_iterator(
        #         "/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5",
        #         "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed",
        #         None,
        #     )
        #     df = pd.read_csv(
        #         "ANC/Selection_Summary_Statistics_01OCT2025.tsv", comment="#", sep="\t"
        #     )
        #     df = df.set_index(["CHROM", "POS"])
        #     for chrom in range(1, 23):
        #         bound = it._chrom_bounds(chrom)
        #         chrom_df = df.xs(chrom, level="CHROM")
        #         chrom_df = chrom_df[~chrom_df.index.duplicated()]
        #         for i in range(0, 500_000_000, 50000):
        #             pos = it.find(i, chrom)
        #             if pos > bound[1]:
        #                 break

        #             region = it.real_region(
        #                 neg1=False, region_len=True, start_idx=pos, return_all_pos=True
        #             )
        #             if region is None:
        #                 continue

        #             region, positions, _ = region

        #             xs = chrom_df["S"].reindex(positions).abs().fillna(-100)

        #             sample = tokenizer(region)
        #             yield {
        #                 "input_ids": sample[..., 0],
        #                 "distances": sample[0, :, 1],
        #                 "chrom": chrom,
        #                 "positions": positions,
        #                 "label": xs,
        #             }

        # features = make_features(
        #     label_dtype="float16", label_resolution="snp", include_snp_pos=True
        # )
        # # Save tokenized data
        # dataset = Dataset.from_generator(gen, features=features)
        # dataset.save_to_disk("ANC/tokenized_CHB")

    elif mode == "fasternn":
        tokenizer = Tokenizer(max_haps=256, num_snps=128)
        split = "test"

        def gen():
            samples = np.load(f"FASTER_NN/fasternn_{split}_regions_512snps.npy")
            distances = np.load(f"FASTER_NN/fasternn_{split}_distances_512snps.npy")
            labels = pd.read_csv(f"FASTER_NN/fasternn_{split}_meta.csv")["label"].values
            for sample, distance in zip(samples, distances):
                region = np.dstack(
                    [sample, distance[None, :].repeat(sample.shape[0], axis=0)]
                )
                region, distances = tokenizer(region)
                yield {"input_ids": region, "distances": distances, "labels": labels}

        # Save tokenized data
        dataset = Dataset.from_generator(
            gen, features=make_features(label_dtype="int8", label_resolution="window")
        )
        dataset.save_to_disk(f"FASTER_NN/tokenized_{split}_512snps")
    elif mode == "runsel":
        which = sys.argv[2]
        allsamples: np.ndarray = np.load(
            f"data/matrices/{which}/matrices.npy", mmap_mode="r"
        )
        alldistances: np.ndarray = np.load(
            f"data/matrices/{which}/distances.npy", mmap_mode="r"
        )
        df = pd.read_csv(f"data/matrices/{which}/metadata.csv")

        low_sel_samples = np.load("data/matrices/pan_4_low_s/matrices.npy", mmap_mode="r")
        low_sel_distances = np.load("data/matrices/pan_4_low_s/distances.npy", mmap_mode="r")
        low_sel_df = pd.read_csv("data/matrices/pan_4_low_s/metadata.csv")
        
        meta_dict = {
            "N1": "float16",
            "N2": "float16",
            "T1": "float16",
            "T2": "float16",
            "growth": "float16",
            "has_dfe": "int8",
            "low_mut": "int8",
            "onset_time": "float16",
            "min_freq": "float16",
        }

        if which == "pan_3":
            mask = df["demo_id"] == int(sys.argv[3])
            allsamples = allsamples[mask]
            alldistances = alldistances[mask]
            df = df[mask].reset_index(drop=True)

        tokenizer = Tokenizer(max_haps=200, num_snps=512)

        features = make_features(
            tokenizer=tokenizer,
            label_dtype="int8",
            label_resolution="window",
            include_s=True,
            extra_features=meta_dict,
        )
        # out of 1000 total, 800 can be used for trainin
        neutrals = df["coeff"] == 0
        selections = df["coeff"] > 0

        train_samples = list(range(800))
        test_samples = list(range(800, 1000))

        if which == "pan_3":
            neutral_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[neutrals],
                    alldistances[neutrals],
                    df[neutrals],
                    times=1,
                    force_s=None,
                ),
                features=features,
            )
            selected_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[selections],
                    alldistances[selections],
                    df[selections],
                    times=1,
                    force_s=True,
                ),
                features=features,
            )
            dataset = concatenate_datasets([neutral_dataset, selected_dataset])
            dataset.save_to_disk(f"data/dataset/{which}_demoid-{sys.argv[3]}_balanced/")
            sys.exit(0)

        for split, name in zip(
            [train_samples, test_samples],
            ["train", "test"],
        ):
            neutral_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[neutrals][split],
                    alldistances[neutrals][split],
                    df[neutrals].iloc[split],
                    times=5,
                    force_s=None,
                    meta_dict=meta_dict
                ),
                features=features,
            )
            selected_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[selections][split],
                    alldistances[selections][split],
                    df[selections].iloc[split],
                    times=10,
                    force_s=True,
                    meta_dict=meta_dict
                ),
                features=features,
            )
            shoulders_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[selections][split],
                    alldistances[selections][split],
                    df[selections].iloc[split],
                    times=5,
                    force_s=False,
                    meta_dict=meta_dict
                ),
                features=features,
            )
            low_s_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    low_sel_samples[split],
                    low_sel_distances[split],
                    low_sel_df.iloc[split],
                    times=5,
                    force_s=True,
                    meta_dict=meta_dict
                ),
                features=features,
            )
            low_s_shoulders_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    low_sel_samples[split],
                    low_sel_distances[split],
                    low_sel_df.iloc[split],
                    times=5,
                    force_s=False,
                    meta_dict=meta_dict
                ),
                features=features,
            )
            
            # shoulders_dataset = Dataset.from_generator(
            #     lambda: gen_sel(
            #         allsamples[selections][split],
            #         alldistances[selections][split],
            #         df[selections].iloc[split],
            #         times=4,
            #         force_s=False,
            #         pos=125000
            #     ),
            #     features=features,
            # )
            dataset = concatenate_datasets(
                [
                    neutral_dataset,
                    selected_dataset,
                    shoulders_dataset,
                    low_s_dataset,
                    low_s_shoulders_dataset,
                ]
            )

            # plot distribution of s
            import matplotlib.pyplot as plt
            s_values = dataset["s"]
            plt.hist(s_values, bins=50)
            plt.xlabel("Selection coefficient (s)")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of selection coefficients in {which} {name} dataset")
            plt.savefig(f"figs/{which}_{name}_s_distribution.png") 
            plt.close()

            dataset.save_to_disk(f"data/dataset/{which}_{name}_with_low_s/")
    elif mode == "runsel_pops":
        which = "combined"
        allsamples: np.ndarray = np.load(
            f"data/matrices/{which}/matrices.npy", mmap_mode="r"
        )
        alldistances: np.ndarray = np.load(
            f"data/matrices/{which}/distances.npy", mmap_mode="r"
        )
        df = pd.read_csv(f"data/matrices/{which}/metadata.csv")

        # shuffle samples, distances, df
        rng = np.random.default_rng(0)
        perm = rng.permutation(allsamples.shape[0])
        allsamples = allsamples[perm]
        alldistances = alldistances[perm]
        df = df.iloc[perm].reset_index(drop=True)

        tokenizer = Tokenizer(max_haps=200, num_snps=512)

        features = make_features(
            tokenizer=tokenizer,
            label_dtype="int8",
            label_resolution="window",
            include_s=True,
        )

        selections = df["coeff"] > 0
        neutrals = df["coeff"] == 0
        n = min(allsamples[neutrals].shape[0], allsamples[selections].shape[0])
        train_samples = list(range(int(n * 0.8)))
        test_samples = list(range(int(n * 0.8), n))

        for split, name in zip(
            [train_samples, test_samples],
            ["train", "test"],
        ):
            neutral_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[neutrals][split],
                    alldistances[neutrals][split],
                    df[neutrals].iloc[split],
                    times=2,
                    force_s=None,
                    start_at=0
                ),
                features=features,
            )
            selected_dataset = Dataset.from_generator(
                lambda: gen_sel(
                    allsamples[selections][split],
                    alldistances[selections][split],
                    df[selections].iloc[split],
                    times=2,
                    force_s=None,
                    start_at=0
                ),
                features=features,
            )
            dataset = concatenate_datasets(
                [
                    neutral_dataset,
                    selected_dataset,
                ]
            )
            dataset.save_to_disk(f"data/dataset/{which}_{name}/")
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
            for _ in range(10000):
                start_bp = np.random.randint(0, last_pos - 64)
                start_idx = int(np.searchsorted(cum_pos, start_bp, side="left"))
                length = rng.integers(low=16, high=64)
                # length = 64
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

        features = make_features(
            label_dtype="float16", label_resolution="window", include_snp_pos=True
        )
        dataset = Dataset.from_generator(
            lambda: gen(
                "1000g/regiontest/ghist_const4.trees",
                "1000g/regiontest/ghist_const4.out",
            ),
            features=features,
        )
        dataset2 = Dataset.from_generator(
            lambda: gen(
                "1000g/regiontest/ghist_const6.trees",
                "1000g/regiontest/ghist_const6.out",
            ),
            features=features,
        )
        dataset = concatenate_datasets([dataset, dataset2])
        dataset.save_to_disk("dataset/selbin_bigregion_snps")
