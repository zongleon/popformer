"""
Utilities for generating Huggingface datasets for haplotype windows.
"""

from collections.abc import Generator
import os
import argparse
import numpy as np
from datasets import Dataset, Features, Array2D, Value, List, concatenate_datasets
from tqdm import tqdm
import tskit
import allel

from pg_gan import global_vars
from pg_gan.real_data_random import RealDataRandomIterator

BOS_TOKEN = 2
EOS_TOKEN = 3
MASK_TOKEN = 4
PAD_TOKEN = 5
NUM_SNPS = 512
MAX_HAPS = 256

def get_iterator(
    h5_file: str, bed_file: str | None, seed: int | None
) -> RealDataRandomIterator:
    iterator = RealDataRandomIterator(filename=h5_file, bed_file=bed_file, seed=seed)

    print(f"Loaded iterator: ({seed})")
    print(f"{iterator.num_snps} SNPs | {iterator.num_samples} samples")

    return iterator


def get_pos_and_dist_vec(ts, snps_total):
    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total

    dist_vec = [0] + [(positions[j + 1] - positions[j]) for j in range(snps_total - 1)]
    return positions, np.array(dist_vec)


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

    haps = np.hstack([bos_vec, sample[:, :n_snps, 0], eos_vec]).astype(np.int8)
    dists = np.hstack([zeros_vec, sample[:, :n_snps, 1], zeros_vec]).astype(np.float16)

    if n_pad_snps > 0:
        pad_vec = np.full((n_haps, n_pad_snps), PAD_TOKEN)
        zeros_pad_vec = np.zeros((n_haps, n_pad_snps))
        haps = np.hstack([haps, pad_vec])
        dists = np.hstack([dists, zeros_pad_vec])

    if n_pad_haps > 0:
        pad_vec = np.full((n_pad_haps, global_vars.NUM_SNPS + 2), PAD_TOKEN)
        haps = np.vstack([haps, pad_vec])

    return haps, dists[0]


def make_features(
    label_dtype: str | None = None,
    label_resolution: str = None,
    include_pop: bool = False,
    include_pos: bool = False,
    include_snp_pos: bool = False,
):
    features = {
        "input_ids": Array2D((MAX_HAPS, global_vars.NUM_SNPS + 2), "int8"),
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
        features["chrom"] = Value("int8")

    if label_dtype is not None:
        if label_resolution == "window":
            features["label"] = Value(label_dtype)
        elif label_resolution == "snp":
            features["label"] = List(Value(label_dtype))
        elif label_resolution == "snphap":
            features["label"] = Array2D((MAX_HAPS, NUM_SNPS + 2), label_dtype)
        else:
            raise ValueError(
                "Invalid label resolution"
                "Supported options are ['window', 'snp', 'snphap']"
            )

    return Features(features)


def find_nonzero_block_cols(sample: np.ndarray) -> tuple[int, int]:
    """
    Find the first and last col indices in a 2D array that are not all zeros.
    Returns (first_idx, last_idx), inclusive.
    If all cols are zero, returns (None, None).
    """
    assert len(sample.shape) == 2, "sample should be a 2D array of (haps, snps)"
    # sample: shape (n_rows, n_cols)
    nonzero_mask = ~(np.all(sample == 0, axis=0))
    nonzero_indices = np.nonzero(nonzero_mask)[0]
    if nonzero_indices.size == 0:
        return (None, None)
    return (nonzero_indices[0], nonzero_indices[-1])


def parse_hdf5_random(filepath: str, args) -> tuple[Generator, Features]:
    # shuffle means get random chromosomal windows
    def gen():
        # use pg gan iterator to get regions
        it = get_iterator(filepath, args.bed_file, args.seed)
        for _ in range(args.n_samples):
            sample = it.real_region(neg1=False, region_len=True)
            region, distance = tokenizer(sample)
            yield {
                "input_ids": region,
                "distances": distance,
            }
    features = make_features()
    return gen, features


def parse_hdf5_inorder(filepath: str, args) -> tuple[Generator, Features]:
    # we're not shuffling: get windows in order from all chroms
    # or user specified chroms
    if args.chrom is not None:
        chroms = [args.chrom]
    else:
        chroms = list(range(1, 23))

    def gen():
        # use pg gan iterator to get regions
        it = get_iterator(filepath, args.bed_file, args.seed)
        for chrom in chroms:
            bound = it._chrom_bounds(chrom)
            tqdm.write(f"{chrom} | {bound[0]} - {bound[1]}")
            pos, i = 0, 0
            while pos < bound[1]:
                pos = it.find(i, chrom)
                i = i + args.window_jump

                region = it.real_region(
                    neg1=False, region_len=True, start_idx=pos, return_pos=True
                )
                if region == "end_chrom":
                    break
                if region is None:
                    continue

                region, s, e, c = region
                region, distance = tokenizer(region)
                
                yield {
                    "input_ids": region,
                    "distances": distance,
                    "start_pos": s,
                    "end_pos": e,
                    "chrom": c,
                }

    features = make_features(include_pos=True)
    return gen, features


def parse_hdf5(filepath: str, args) -> tuple[Generator, Features]:
    
    # parse_file should always return a generator function
    if args.shuffle:
        return parse_hdf5_random(filepath, args)
    else:
        return parse_hdf5_inorder(filepath, args)


def parse_trees_random(filepath: str, args) -> tuple[Generator, Features]:
    rng = np.random.default_rng(args.seed)
    def gen():
        # process tree
        ts = tskit.load(filepath)
        gt_matrix = ts.genotype_matrix()
        num_snps = gt_matrix.shape[0]
        positions, dist_vec = get_pos_and_dist_vec(ts, num_snps)

        gt_matrix = gt_matrix.T

        cum_pos = np.cumsum(dist_vec)

        last_pos = int(cum_pos[-1])
        start_bp = 0
        for _ in range(args.n_samples):
            start_bp = np.random.randint(0, last_pos - 64)
            start_idx = int(np.searchsorted(cum_pos, start_bp, side="left"))
            # TODO which arg controls how long this should be
            length = rng.integers(low=16, high=64)
            end_idx = start_idx + length

            m = gt_matrix[:, start_idx:end_idx]
            d = dist_vec[start_idx:end_idx].copy()
            p = positions[start_idx:end_idx]
            d[0] = 0

            dist = d[None, :].repeat(m.shape[0], axis=0)
            region = np.dstack([m, dist])
            sample = tokenizer(region)

            yield {
                "input_ids": sample[..., 0],
                "distances": sample[0, :, 1],
                "chrom": 0,
                "positions": p,
            }

            start_bp = start_bp + args.window_jump

    features = make_features(include_snp_pos=True)
    return gen, features


def parse_trees_inorder(filepath: str, args) -> tuple[Generator, Features]:
    def gen():
        # process tree
        ts = tskit.load(filepath)
        gt_matrix = ts.genotype_matrix()
        num_snps = gt_matrix.shape[0]
        positions, dist_vec = get_pos_and_dist_vec(ts, num_snps)

        gt_matrix = gt_matrix.T

        # 50 kbp windows based on cumulative physical distance
        cum_pos = np.cumsum(dist_vec)

        last_pos = int(cum_pos[-1])
        start_bp = 0
        while start_bp <= last_pos:
            end_bp = start_bp + args.window_size
            start_idx = int(np.searchsorted(cum_pos, start_bp, side="left"))
            end_idx = int(np.searchsorted(cum_pos, end_bp, side="left"))

            m = gt_matrix[:, start_idx:end_idx]
            d = dist_vec[start_idx:end_idx].copy()
            p = positions[start_idx:end_idx]
            d[0] = 0

            dist = d[None, :].repeat(m.shape[0], axis=0)
            region = np.dstack([m, dist])
            sample = tokenizer(region)

            yield {
                "input_ids": sample[..., 0],
                "distances": sample[0, :, 1],
                "chrom": 0,
                "positions": p,
            }

            start_bp = start_bp + args.window_jump

    features = make_features(include_snp_pos=True)
    return gen, features

def parse_trees(filepath: str, args) -> tuple[Generator, Features]:
    if args.shuffle:
        return parse_trees_random(filepath, args)
    else:
        return parse_trees_inorder(filepath, args)

def parse_file(filepath, args):
    if os.path.isdir(filepath):
        return None, None
    ext = os.path.splitext(filepath)[1]
    if ext == ".vcf":
        # convert to h5
        newfile = filepath.replace(".vcf", ".h5")
        allel.vcf_to_hdf5(filepath, newfile, fields=["CHROM", "GT", "POS"])
        ext = ".h5"
        filepath = newfile

    if ext == ".h5":
        return parse_hdf5(filepath, args)
    elif ext == ".trees":
        return parse_trees(filepath, args)
    else:
        # non-.vcf, .h5 filetype
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an input file/directory.")
    parser.add_argument(
        "input",
        help="Input file or directory. Supported filetypes: "
        "- vcf (converted to hdf5)"
        "- hdf5"
        "- txt (in ms format)"
        "- .trees (output from stdpopsim)"
        "- directories of any of the above formats",
    )
    parser.add_argument(
        "output",
        help="Path to output. Directories will be created if not existing, and "
        "overwritten if existing. Output will be in a huggingface dataset.",
    )
    parser.add_argument(
        "--bed_file", type=str, default=None, help="Optional BED file for masking."
    )
    parser.add_argument(
        "--chrom", type=int, default=None, help="Chromosome to process. Default is all human autosomes (1-22)."
    )
    parser.add_argument(
        "--L", type=int, default=50000, help="Window size"
    )
    parser.add_argument(
        "--S", type=int, default=512, help="Number of SNPs"
    )
    parser.add_argument(
        "--window_jump", type=int, default=50000, help="Window jump size."
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Whether to shuffle samples"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100000, help="Number of samples to draw if shuffling."
    )
    args = parser.parse_args()

    global_vars.L = args.L
    global_vars.NUM_SNPS = args.S

    if os.path.isdir(args.input):
        # directory
        files = os.listdir(args.input)
        datasets = []
        for file in files:
            filepath = os.path.join(args.input, file)
            gen, features = parse_file(filepath, args)
            if gen is None:
                print(f"Skipping unsupported file: {filepath}")
                continue
            datasets.append(Dataset.from_generator(gen, features=features))

        dataset = concatenate_datasets(datasets)
            
    else:
        # single file
        gen, features = parse_file(args.input, args)
        dataset = Dataset.from_generator(gen, features=features)
        
    dataset.save_to_disk(args.output)


    