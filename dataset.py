"""
Utilities for a dataset.
"""

import argparse
import os

import numpy as np
from datasets import Dataset, Features, Array2D, Value, List
from tqdm import tqdm

from pg_gan import global_vars
from pg_gan.generator import Generator
from pg_gan.global_vars import DEFAULT_SEED, NUM_SNPS
from pg_gan.real_data_random import RealDataRandomIterator
from pg_gan.ss_helpers import parse_output
from pg_gan.util import parse_args, process_opts

VERSION = 2

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


def get_iterator(pop: str, seed=None, custom_bed=None) -> RealDataRandomIterator:
    h5_file = GENOME_PATH.format(pop=pop)
    bed_file = BED_PATH if custom_bed is None else custom_bed
    s = seed if seed is not None else DEFAULT_SEED
    iterator = RealDataRandomIterator(filename=h5_file, bed_file=bed_file, seed=s)

    print(f"Loaded iterator: {pop} ({s})")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_iterator_ghist(name: str, bed_name: str) -> RealDataRandomIterator:
    iterator = RealDataRandomIterator(filename=name, bed_file=bed_name, seed=0)

    print(f"Loaded iterator: {name}")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_data(pop: str, n_samples: int, seed=None) -> np.ndarray:
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

    for i in tqdm(range(n_samples)):
        sample = iterator.real_region(neg1=False, region_len=True)        
        samples[i] = tokenizer(sample)

    return samples

def tokenizer(sample: np.ndarray) -> np.ndarray:
    # remove -1s if present, scale distances
    sample[..., 0][sample[..., 0] == -1] = 0
    sample[..., 1] = sample[..., 1] * global_vars.L

    # padding
    n_haps = min(sample.shape[0], MAX_HAPS)
    n_snps = min(sample.shape[1], NUM_SNPS)
    n_pad_haps = MAX_HAPS - n_haps
    n_pad_snps = NUM_SNPS - n_snps

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
        pad_vec = np.full((n_pad_haps, NUM_SNPS + 2), PAD_TOKEN)
        zeros_pad_vec = np.zeros((n_pad_haps, NUM_SNPS + 2))
        haps = np.vstack([haps, pad_vec])
        dists = np.vstack([dists, zeros_pad_vec])

    return np.dstack([haps, dists])


def make_features(
    label_dtype: str | None = None,
    label_resolution: str = None,
    include_pop: bool = False,
    include_pos: bool = False
):
    features = {
        "input_ids": Array2D((256, NUM_SNPS + 2), "int8"),
        "distances": List(Value("float32")),
    }
    if include_pop:
        features["pop"] = Value(dtype="string")
    if include_pos:
        features["start_pos"] = Value(dtype="int32")
        features["end_pos"] = Value(dtype="int32")
        features["chrom"] = Value("int8")

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
        choices=[
            "gen",
            "runsimple",
            "runrealsim",
            "runsel",
            "ghist",
            "lai",
            "realsel"
        ],
        help="Mode to run",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument("--extra", type=str, default="")
    args = parser.parse_args()

    mode = args.mode
    n_samples = args.n_samples

    if mode == "gen":
        for pop in ["CEU", "CHB", "YRI"]:
            samples = get_data(pop=pop, n_samples=n_samples, seed=0)
            save_data(pop, samples)
    elif mode == "runsimple":
        def gen():
            for pop in ["CEU", "CHB", "YRI"]:
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
        dataset.save_to_disk(f"dataset{VERSION}/tokenizedrealsim")
    elif mode == "runsel":
        def gen():
            for t, sel, sel_bin in zip(
                ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
                [0, 0.1, 0.01, 0.05, 0.025],
                [0, 1, 1, 1, 1]
            ):
                for pop in ["CEU", "CHB", "YRI"]:
                    samples: np.ndarray = np.load(f"1000g/{pop}/matrices_regions_{pop}_{t}.npy")[:2400]
                    distances: np.ndarray = np.load(f"1000g/{pop}/distances_regions_{pop}_{t}.npy")[:2400]

                    for sample, dist in zip(samples, distances):
                        sample = sample.T
                        first, last = find_nonzero_block_cols(sample)
                        sample = np.dstack([
                            sample[:, first:last],
                            dist[None, first:last].repeat(sample.shape[0], axis=0)
                        ])
                        sample = tokenizer(sample)

                        yield {
                            "input_ids": sample[..., 0],
                            "distances": sample[0, :, 1],
                            "label": sel_bin,
                        }
                        

        features = make_features(label_dtype="float32", label_resolution="window")
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk(f"dataset{VERSION}/tokenizedsel2")
    elif mode == "ghist":
        t = args.extra
        def gen():
            it = get_iterator_ghist(
                f"GHIST/process/GHIST_2025_{t}.21.testing_process.h5", "GHIST/raw/21.accessible.bed"
            )
            n_snps = it.num_snps

            for i in range(0, n_snps, 16):
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
        dataset.save_to_disk(f"GHIST/ghist_samples_{t}")
    
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
        t = args.extra
        def gen():
            it = get_iterator("CEU", 0, "SEL/bed.bed")
            n_snps = it.num_snps
            n_yielded = 0
            for i in range(0, n_snps, 1):
                if n_yielded > n_samples:
                    break
                region = it.real_region(neg1=False, region_len=True, start_idx=i, return_pos=True)
                if region is not None:
                    region, s, e, c = region
                    sample = tokenizer(region)
                    n_yielded += 1
                    yield {
                        "input_ids": sample[..., 0], 
                        "distances": sample[0, :, 1], "start_pos": s, "end_pos": e,
                        "chrom": c
                    }

        features = make_features(include_pos=True)
        # Save tokenized data
        dataset = Dataset.from_generator(gen, features=features)
        dataset.save_to_disk("SEL/tokenized_CEU")
