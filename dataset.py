"""
Utilities for a dataset.
"""

import argparse
import os

import numpy as np
from datasets import Dataset, Features, Array2D, Value, List, DatasetDict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from pg_gan import global_vars
from pg_gan import util
from pg_gan.generator import Generator
from pg_gan.global_vars import DEFAULT_SEED, NUM_SNPS
from pg_gan.real_data_random import RealDataRandomIterator
from pg_gan.ss_helpers import parse_output
from pg_gan.util import parse_args, process_opts

VERSION = 4

OUTFILE_PATH = "outfiles/{pop}/{pop}_{seed}_{model}.out"
GENOME_PATH = "/bigdata/smathieson/1000g-share/HDF5/{pop}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
BED_PATH = "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed"

BOS_TOKEN = 2
EOS_TOKEN = 3
PAD_TOKEN = 5
MAX_HAPS = 256

pop = None

_OUTPUT_SAMPLES = f"dataset{VERSION}/samples.npz"


def read_outfile(file: str) -> Generator:
    """
    Load parameters from a file, returns a generator with those params.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")

    # condensed output file parsing for options. don't print
    param_values, in_file_data = parse_output(file)
    opts, param_values = parse_args(
        in_file_data=in_file_data, param_values=param_values
    )
    generator, _, _, _ = process_opts(opts, summary_stats=True)
    generator.update_params(param_values)

    print("Loaded generator:")
    print(generator.curr_params)

    return generator


def get_iterator(pop: str, seed=None, custom_bed=None, use_bed=True) -> RealDataRandomIterator:
    h5_file = GENOME_PATH.format(pop=pop)
    bed_file = BED_PATH if custom_bed is None else custom_bed
    s = seed if seed is not None else DEFAULT_SEED
    iterator = RealDataRandomIterator(filename=h5_file, 
                                      bed_file=bed_file if use_bed else None, 
                                      seed=s)

    print(f"Loaded iterator: {pop} ({s})")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_iterator_ghist(name: str, bed_name: str) -> RealDataRandomIterator:
    iterator = RealDataRandomIterator(filename=name, bed_file=bed_name, seed=0)

    print(f"Loaded iterator: {name}")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_data(pop: str, n_samples: int, seed=None, snp=False) -> np.ndarray:
    """
    Get a dataset of real and simulated data.

    Returns a tuple of (samples, labels, sources):
        Samples is a numpy array of shape (n_samples, num_snps, 2).
    """
    iterator = get_iterator(pop=pop, seed=seed)

    # get n_samples from each iterator
    samples = np.zeros(
        (n_samples, MAX_HAPS, NUM_SNPS + 2, 2), dtype=np.float32
    )
    # Collect the lengths for histogram
    lengths = []
    for i in tqdm(range(n_samples)):
        if snp:
            global_vars.NUM_SNPS = np.random.randint(32, NUM_SNPS)
        sample = iterator.real_region(neg1=False, region_len=(not snp))
        lengths.append(sample.shape[1])
        samples[i] = tokenizer(sample)

    # Print histogram of sample.shape[1]
    bins = np.linspace(0, 1024, 32)
    plt.hist(lengths, bins=bins, label=pop, alpha=0.5)
    plt.xlabel("Number of SNPs (sample.shape[1])")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Histogram of SNP counts, window size {global_vars.L}")
    if pop == "GBR" and not snp:
        plt.savefig(f"figs/snp_hist_{global_vars.L}.png", dpi=300)

    return samples

def tokenizer(sample: np.ndarray) -> np.ndarray:
    # remove -1s if present, scale distances
    sample[..., 0][sample[..., 0] == -1] = 0
    sample[..., 1] = sample[..., 1] * global_vars.L

    # padding
    n_haps = min(sample.shape[0], MAX_HAPS)
    n_snps = min(sample.shape[1], global_vars.NUM_SNPS)
    n_pad_haps = MAX_HAPS - n_haps
    n_pad_snps = global_vars.NUM_SNPS - n_snps

    # start and end tokens
    bos_vec = np.full((n_haps, 1), BOS_TOKEN)
    eos_vec = np.full((n_haps, 1), EOS_TOKEN)
    zeros_vec = np.zeros((n_haps, 1))

    haps = np.hstack([bos_vec, sample[:, :n_snps, 0], eos_vec])
    dists = np.hstack([zeros_vec, sample[:, :n_snps, 1], zeros_vec])

    if n_pad_snps > 0:
        pad_vec = np.full((n_haps, n_pad_snps), PAD_TOKEN)
        zeros_pad_vec = np.zeros((n_haps, n_pad_snps))
        haps = np.hstack([haps, pad_vec])
        dists = np.hstack([dists, zeros_pad_vec])

    if n_pad_haps > 0:
        pad_vec = np.full((n_pad_haps, global_vars.NUM_SNPS + 2), PAD_TOKEN)
        zeros_pad_vec = np.zeros((n_pad_haps, global_vars.NUM_SNPS + 2))
        haps = np.vstack([haps, pad_vec])
        dists = np.vstack([dists, zeros_pad_vec])

    return np.dstack([haps, dists])


def make_features(
    label_dtype: str | None = None,
    label_resolution: str = None,
    include_pop: bool = False,
    include_pos: bool = False,
    include_snp_pos: bool = False,
):
    features = {
        "input_ids": Array2D((256, global_vars.NUM_SNPS + 2), "int8"),
        "distances": List(Value("float32")),
    }
    if include_pop:
        features["pop"] = Value(dtype="string")
    if include_pos:
        features["start_pos"] = Value(dtype="int32")
        features["end_pos"] = Value(dtype="int32")
        features["chrom"] = Value("int8")
    if include_snp_pos:
        features["positions"] = List(Value(dtype="int32"))

    if label_dtype is not None:
        if label_resolution == "window":
            features["label"] = Value(label_dtype)
        elif label_resolution == "snp":
            features["label"] = List(Value(label_dtype))
        elif label_resolution == "snphap":
            features["label"] = Array2D((256, NUM_SNPS + 2), label_dtype)
        else:
            raise ValueError(
                "Invalid label resolution"
                "Supported options are ['window', 'snp', 'snphap']"
            )

    return Features(features)


def save_data(pop: str, samples: np.ndarray):
    """
    Save the dataset to npz and csv files.
    """
    d = os.path.dirname(_OUTPUT_SAMPLES)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    np.savez_compressed(os.path.join(d, f"X_{pop}.npz"), X=samples)
    print(f"Saved {len(samples)} samples with {samples.shape[2]} SNPs each.")


def load_data(pop: str, dir=None) -> np.ndarray:
    """
    Load the dataset from the npz file.
    Returns a tuple of (samples, labels).
    """
    if dir:
        file = os.path.join(dir, "samples.npz")
    else:
        file = _OUTPUT_SAMPLES
    d = os.path.dirname(file)
    if not os.path.exists(d):
        raise FileNotFoundError(f"Dataset path {file} does not exist.")

    # Load samples and labels into memory
    samples = np.load(os.path.join(d, f"X_{pop}.npz"))["X"]

    print(f"Loaded {len(samples)} samples with {samples.shape[2]} SNPs each.")

    return samples


def find_nonzero_block_cols(sample: np.ndarray) -> tuple[int, int]:
    """
    Find the first and last col indices in a 2D array that are not all zeros.
    Returns (first_idx, last_idx), inclusive.
    If all cols are zero, returns (None, None).
    """
    # sample: shape (n_rows, n_cols)
    nonzero_mask = ~(np.all(sample == 0, axis=0))
    nonzero_indices = np.where(nonzero_mask)[0]
    if nonzero_indices.size == 0:
        return (None, None)
    return (nonzero_indices[0], nonzero_indices[-1])


if __name__ == "__main__":
    # "Usage: python dataset.py <gen | tokenize> [n_samples] pop")
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument(
        "mode",
        help="Mode to run",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument("--extra", type=str, default="")
    parser.add_argument("--extra2", type=str, default="")
    args = parser.parse_args()

    mode = args.mode
    n_samples = args.n_samples

    if mode == "gen":
        if args.extra == "genomewindow":
            for pop in ["CEU", "CHB", "YRI", "CHS", "ESN", "GBR"]:
                global_vars.L = 50000
                samples = get_data(pop=pop, n_samples=n_samples, seed=0)
                save_data(pop, samples)
        elif args.extra == "snpwindow":
            for pop in ["CEU", "CHB", "YRI", "CHS", "ESN", "GBR"]:
                global_vars.NUM_SNPS = 512
                samples = get_data(pop=pop, n_samples=n_samples, seed=0, snp=True)
                save_data(pop, samples)

    elif mode == "runsimple":
        def gen():
            for pop in ["CEU", "CHB", "YRI", "CHS", "ESN", "GBR"]:
                samples = load_data(pop)
                for sample in samples:
                    yield {
                        "input_ids": sample[..., 0],
                        "distances": sample[0, :, 1],  # all haps same dists
                        "pop": pop,
                    }

        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=make_features(include_pop=True))
        dataset.save_to_disk(f"dataset{VERSION}/tokenized")

    elif mode == "runrealsim":
        def gen():
            for pop in ["CEU", "CHB", "YRI"]:
                samples = np.load(f"../disc-interpret/dataset-{pop}/X.npy")
                labels = np.load(f"../disc-interpret/dataset-{pop}/y.npy")

                subset = np.random.choice(samples.shape[0], n_samples, replace=False)
                samples = samples[subset]
                labels = labels[subset]

                for sample, label in zip(samples, labels):
                    sample = tokenizer(sample)
                    yield {
                        "input_ids": sample[..., 0],
                        "distances": sample[0, :, 1],
                        "label": label,
                    }

        features = make_features(label_dtype="int8", label_resolution="window")
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk(f"dataset{VERSION}/ft_realsim_tkns")
    elif mode == "runsel":
        allsamples: np.ndarray = np.load("1000g/combined_pan_2/matrices.npy", mmap_mode="r")
        alldistances: np.ndarray = np.load("1000g/combined_pan_2/distances.npy", mmap_mode="r")
        df = pd.read_csv("1000g/combined_pan_2/metadata.csv")
        filt = None
        global_vars.NUM_SNPS = 512

        def gen():
            sel = df["coeff"].to_numpy()            
            sel = (sel > 0).astype(int)

            for i, (sample, dist, s) in enumerate(zip(allsamples, alldistances, sel)):
                if not filt[i]:
                    continue
                sample = np.dstack([
                    sample,
                    dist[None, :].repeat(sample.shape[0], axis=0)
                ])
                sample = tokenizer(sample)

                yield {
                    "input_ids": sample[..., 0],
                    "distances": sample[0, :, 1],
                    "label": s,
                }
                    
        features = make_features(label_dtype="int8", label_resolution="window")
        # Save tokenized data
        filt = df["pop"] == "pan_2"
        train_dataset = Dataset.from_generator(gen, features=features)
        filt = df["pop"] == "YRI"
        test_dataset = Dataset.from_generator(gen, features=features)
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })
        dataset.save_to_disk(f"dataset{VERSION}/ft_selbin_fixwindow_tkns")
    elif mode == "ghist":
        global_vars.L = 50000
        global_vars.NUM_SNPS = 64
        t = args.extra
        def gen():
            it = get_iterator_ghist(
                f"GHIST/process/GHIST_2025_{t}.21.testing_process.h5", "GHIST/raw/21.accessible.bed"
            )
            n_snps = it.num_snps

            for i in range(0, n_snps, 32):
                region = it.real_region(neg1=False, region_len=True, start_idx=i, return_pos=True)
                if region is not None:
                    region, s, e, c = region
                    sample = tokenizer(region)
                    yield {
                        "input_ids": sample[..., 0], 
                        "distances": sample[0, :, 1], "start_pos": s, "end_pos": e,
                        "chrom": c
                    }

        features = make_features(include_pos=True)
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk(f"GHIST/samples_{t}_{global_vars.L}")
    
    elif mode == "lai":
        t = args.extra
        def gen():
            it = get_iterator("CEU", 0)
            n_snps = it.num_snps

            for i in range(0, n_snps, 64):
                if i / 64 > n_samples:
                    break
                region = it.real_region(neg1=False, region_len=True, start_idx=i, return_pos=True)
                if region is not None:
                    region, s, e, c = region
                    sample = tokenizer(region)
                    yield {
                        "input_ids": sample[..., 0], 
                        "distances": sample[0, :, 1], "start_pos": s, "end_pos": e,
                        "chrom": c
                    }

        features = make_features(include_pos=True)
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk("LAI/tokenized")
    elif mode == "realsel":
        pop = args.extra
        global_vars.NUM_SNPS = 512
        global_vars.L = 100000
        def gen():
            it = get_iterator(pop, 0, None, use_bed=False) #"SEL/chr1.bed")
            for chrom in range(1, 2):
                bound = it._chrom_bounds(chrom)
                tqdm.write(f"Pop {pop} | Chrom {chrom:<2d} | {bound}")
                for i in range(0, 500000000, 100000):
                    pos = it.find(i, chrom)
                    if pos > bound[1]:
                        break

                    region = it.real_region(neg1=False, region_len=True, start_idx=pos, return_pos=True)
                    if region is None:
                        break
                    
                    region, s, e, c = region
                    sample = tokenizer(region)
                    yield {
                        "input_ids": sample[..., 0], 
                        "distances": sample[0, :, 1], "start_pos": s, "end_pos": e,
                        "chrom": c
                    }

        features = make_features(include_pos=True)
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk(f"SEL/tokenized_{pop}")
    elif mode == "fasternn":
        global_vars.NUM_SNPS = 512
        def gen():
            samples = np.load("FASTER_NN/fasternn_regions_majmin512.npy")
            distances = np.load("FASTER_NN/fasternn_distances_majmin512.npy")
            for sample, distance in zip(samples, distances):
                region = np.dstack([
                    sample,
                    distance[None, :].repeat(sample.shape[0], axis=0)
                ])
                region = tokenizer(region)
                yield {
                    "input_ids": region[..., 0],
                    "distances": region[0, :, 1]
                }

        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=make_features())
        dataset.save_to_disk("FASTER_NN/tokenized_majmin512")
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
        it = get_iterator_ghist(
            "IMP/KHV_infmasked_ref.h5",
            None
        )
        it2 = get_iterator_ghist(
            "IMP/KHV_infmasked_tgt.h5",
            None
        )
        n = it2.num_samples
        n_snps = it.num_snps
        n_haps_ref = it.num_samples
        n_haps = it.num_samples + it2.num_samples
        tqdm.write(f"{n_snps}, {n_haps}")

        region = positions = None

        snp_ref = snp_tgt = 0
        while snp_ref < n_snps:
            cur_idx = snp_ref % global_vars.NUM_SNPS
            if cur_idx == 0:
                if snp_tgt != 0:
                    # save if not first
                    region = shuffle(region, n)
                    dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L
                        for j in range(len(positions)-1)]

                    region = util.process_gt_dist(region, dist_vec,
                        region_len=False, real=True, neg1=False)
                    
                    sample = tokenizer(region)

                    samples_list.append({
                        "input_ids": sample[..., 0],
                        "distances": sample[0, :, 1],
                        "positions": positions
                    })

                region = np.zeros((global_vars.NUM_SNPS, n_haps))
                positions = np.zeros((global_vars.NUM_SNPS, ))
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
            dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L
                for j in range(len(positions)-1)]
            region = util.process_gt_dist(region, dist_vec,
                region_len=False, real=True, neg1=False)
            sample = tokenizer(region)
            samples_list.append({
                "input_ids": sample[..., 0],
                "distances": sample[0, :, 1],
                "positions": positions
            })

        features = make_features(include_snp_pos=True)
        # Save tokenized data
        dataset = Dataset.from_list(samples_list, features=features)
        dataset.save_to_disk(f"IMP/infmasked_{global_vars.NUM_SNPS}")
