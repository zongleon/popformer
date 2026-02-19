"""
Some specialty dataloaders for specific datasets.
"""

import glob
import os
import allel
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
from popformer.real_data_random import RealDataRandomIterator


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


def gen_sel(
    samps,
    dists,
    meta,
    meta_dict={},
    times=1,
    force_s=False,
    start_at=None,
    binary=True,
    window=50000,
    resolution="window",
):
    sel = meta["coeff"].to_numpy(dtype=np.float16)
    pos = meta["coordinate"].fillna(-1).to_numpy(dtype=np.int32)

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
                if start_at is not None:
                    if start_at == 0:
                        start_idx = 0
                    else:
                        start_idx = np.searchsorted(positions, start_at)
                else:
                    start_idx = np.random.randint(0, length)
                # get 50k region
                if positions[start_idx] + window > positions[-1]:
                    end_idx = length
                else:
                    end_idx = np.searchsorted(positions, positions[start_idx] + window)
                # end_idx = min(start_idx + region_len, length)
                found_sample = sample[:, start_idx:end_idx]
                found_dist = dist[start_idx:end_idx]

                # check if selection is in region
                # if force_s is True, region must contain selected site
                # if force_s is False, region must be a shoulder region
                # if force_s is None, any region is allowed
                # sel should be 0
                SHOULDER_DISTANCE = 10000 if window > 50000 else 0

                region_start = positions[start_idx]
                region_end = positions[end_idx - 1]

                shoulder = False

                # Shoulders: region must be at least SHOULDER_DISTANCE away from selected site
                if force_s is False:
                    if (
                        abs(p - region_start) < SHOULDER_DISTANCE
                        or abs(region_end - p) < SHOULDER_DISTANCE
                    ):
                        found = False
                        continue
                    if region_start <= p <= region_end:
                        found = False
                        continue
                    found = True
                    shoulder = True

                # Containing: selected site must be in the middle SHOULDER_DISTANCE of the region
                elif force_s:
                    if not (
                        region_start + SHOULDER_DISTANCE // 2
                        <= p
                        <= region_end - SHOULDER_DISTANCE // 2
                    ):
                        found = False
                        continue
                    if not (region_start <= p <= region_end):
                        found = False
                        continue
                    found = True

                # Neutral: any region is allowed
                elif force_s is None:
                    found = True

                # avoid too small regions
                if positions[end_idx - 1] - positions[start_idx] < 10000:
                    found = False

                tqdm.write(
                    f"Sample {i}: pos {positions[start_idx]}-{positions[end_idx - 1]}, s={s}, p={p}"
                    f"\n{start_idx}-{end_idx}, length {length}"
                    f"\n  min: {positions[0]}, max: {positions[-1]}"
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
            # tqdm.write(f"label {1 if s > 0 and not shoulder else 0}, s={s}")

            if resolution == "window":
                # for window-based labeling, label is 1 if selected, 0 if neutral/shoulder
                label = 1 if s > 0 and not shoulder else 0
                label = s if not binary else label
            elif resolution == "snp":
                # for snp-based labeling, label is a list
                # containing 0s for all positions except the selected site (1)
                label = np.zeros(region.shape[1], dtype=np.int8)
                if s > 0 and not shoulder:
                    # find index of selected site in region
                    region_positions = np.cumsum(distances) + positions[start_idx]
                    selected_idx = np.searchsorted(region_positions, p)
                    if (
                        selected_idx < region.shape[1]
                        and abs(region_positions[selected_idx] - p) < 1e4
                    ):
                        label[selected_idx] = 1

            result = {
                "input_ids": region,
                "distances": distances,
                "label": label,
                "s": s,
                "shoulder": int(shoulder),
            }

            for meta_key, dtype in meta_dict.items():
                result[meta_key] = np.array(meta.iloc[i][meta_key])

            yield result


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pt":
        tokenizer = Tokenizer(max_haps=256, num_snps=512)

        # glob files for .vcf.gz, one for each population
        # convert them to hdf
        path = sys.argv[2]
        files = glob.glob(os.path.join(path, "*.vcf.gz"))
        h5_files = []

        for path in files:
            # convert to h5
            print(f"Processing {path}")
            newfile = path.replace(".vcf.gz", ".h5")
            if not os.path.exists(newfile):
                allel.vcf_to_hdf5(
                    path, newfile, fields=["CHROM", "GT", "POS"], overwrite=True
                )
            h5_files.append(newfile)

        def gen():
            rng = np.random.default_rng(0)
            # use pg gan iterator to get region
            # pick a random chrom
            for file in h5_files:
                it = RealDataRandomIterator(file, sys.argv[3])
                for _ in range(5000):
                    while True:
                        chrom = rng.integers(1, 23)

                        pos = rng.integers(*it._chrom_bounds(chrom))
                        i = it.find(pos, chrom)
                        region = it.real_region(
                            tokenizer.num_snps,
                            region_len=True,
                            region_len_size=50_000,
                            start_idx=i,
                            return_pos=True,
                            frac_callable=0.95,
                        )
                        if region != "end_chrom" and region is not None:
                            break

                    region, s, e, c = region
                    region, distance = tokenizer(region)

                    # tqdm.write(
                    #     f"Pop {os.path.basename(file)[:3]}, chrom {c}, pos {pos}, start {s}, end {e}"
                    # )

                    yield {
                        "input_ids": region,
                        "distances": distance,
                        "start_pos": s,
                        "end_pos": e,
                        "chrom": c,
                        "pop": os.path.basename(file)[:3],
                    }

        features = make_features(
            tokenizer=tokenizer, label_dtype=None, include_pos=True, include_pop=True
        )
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk("data/dataset/pt")

    elif mode == "realsim":

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

    elif mode == "runsel":
        which = sys.argv[2]
        allsamples: np.ndarray = np.load(
            f"data/matrices/{which}/matrices.npy", mmap_mode="r"
        )
        alldistances: np.ndarray = np.load(
            f"data/matrices/{which}/distances.npy", mmap_mode="r"
        )
        df = pd.read_csv(f"data/matrices/{which}/metadata.csv")
        generator_df = pd.read_csv("~/projects/onsettofreq/simulate/generators.csv")

        df["pop"] = df.apply(
            lambda row: generator_df[
                (generator_df["N1"] == row["N1"]) & (generator_df["N2"] == row["N2"])
            ]["pop"].iloc[0],
            axis=1,
        )

        meta_dict = {
            "N1": "float16",
            "N2": "float16",
            "T1": "float16",
            "T2": "float16",
            "growth": "float16",
            "mutation_rate": "float16",
            "reco_rate": "float16",
            "has_dfe": "int8",
            # "low_mut": "int8",
            "onset_time": "float16",
            # "min_freq": "float16",
            "goal_freq": "float16",
        }

        tokenizer = Tokenizer(max_haps=200, num_snps=512)

        features = make_features(
            tokenizer=tokenizer,
            label_dtype="int8",
            label_resolution="window",
            # label_resolution="snp",
            include_s=True,
            include_shoulder=True,
            extra_features=meta_dict,
        )

        REPEAT = 1
        window = 50000
        for pop in df["pop"].unique():
            pop_df = df[df["pop"] == pop]
            pop_samples = allsamples[df["pop"] == pop]
            pop_distances = alldistances[df["pop"] == pop]
            neutrals = pop_df["coeff"] == 0
            selections = pop_df["coeff"] > 0

            # use split
            split = 0.8
            total = min(neutrals.sum(), selections.sum())
            print(f"Total samples per class: {total}")

            rng = np.random.default_rng(0)
            train_samples = rng.choice(
                total, size=int(split * total), replace=False
            ).tolist()
            test_samples = list(set(range(total)) - set(train_samples))

            for split, name in zip(
                [train_samples, test_samples],
                ["train", "test"],
            ):
                if split is None:
                    continue
                neutral_dataset = Dataset.from_generator(
                    lambda: gen_sel(
                        pop_samples[neutrals][split],
                        pop_distances[neutrals][split],
                        pop_df[neutrals].iloc[split],
                        times=REPEAT,
                        force_s=None,
                        meta_dict=meta_dict,
                        window=window,
                    ),
                    features=features,
                )
                selected_dataset = Dataset.from_generator(
                    lambda: gen_sel(
                        pop_samples[selections][split],
                        pop_distances[selections][split],
                        pop_df[selections].iloc[split],
                        times=REPEAT * 2,
                        force_s=True,
                        meta_dict=meta_dict,
                        window=window,
                    ),
                    features=features,
                )
                shoulders_dataset = Dataset.from_generator(
                    lambda: gen_sel(
                        pop_samples[selections][split],
                        pop_distances[selections][split],
                        pop_df[selections].iloc[split],
                        times=REPEAT,
                        force_s=False,
                        meta_dict=meta_dict,
                        window=window,
                    ),
                    features=features,
                )
                dataset = concatenate_datasets(
                    [
                        neutral_dataset,
                        selected_dataset,
                        shoulders_dataset,
                    ]
                )

                # dataset = dataset.class_encode_column("label")

                labels = dataset["label"]
                unique, counts = np.unique(labels, return_counts=True)
                label_dist = dict(zip(unique, counts))
                print(f"Label distribution in {which} {name} dataset: {label_dist}")

                # plot distribution of s
                import matplotlib.pyplot as plt

                s_values = dataset["s"]
                plt.hist(s_values, bins=50)
                plt.xlabel("Selection coefficient (s)")
                plt.ylabel("Frequency")
                plt.title(
                    f"Distribution of selection coefficients in {which} {name} dataset"
                )
                plt.savefig(f"figs/{which}_{name}_s_distribution.png")
                plt.close()

                dataset.save_to_disk(f"data/dataset/{which}{pop}_{name}")
