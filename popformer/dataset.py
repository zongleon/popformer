"""
Utilities for generating Huggingface datasets for haplotype windows.
"""

import argparse
import os

import allel
import numpy as np
import tskit
from datasets import Array2D, Dataset, Features, List, Value, concatenate_datasets
from .real_data_random import RealDataRandomIterator
from .util import process_gt_dist
from tqdm import tqdm


def get_pos_and_dist_vec(ts, snps_total):
    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total

    dist_vec = [0] + [(positions[j + 1] - positions[j]) for j in range(snps_total - 1)]
    return positions, np.array(dist_vec)


class Tokenizer:
    BOS_TOKEN = 2
    EOS_TOKEN = 3
    MASK_TOKEN = 4
    PAD_TOKEN = 5

    def __init__(self, max_haps: int, num_snps: int):
        self.max_haps = max_haps
        self.num_snps = num_snps

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return self.tokenizer(sample)

    def get_config(self) -> dict:
        return {
            "max_haps": self.max_haps,
            "num_snps": self.num_snps,
        }

    def tokenizer(self, sample: np.ndarray) -> np.ndarray:
        # padding
        n_haps = min(sample.shape[0], self.max_haps)
        n_snps = min(sample.shape[1], self.num_snps)
        n_pad_haps = self.max_haps - n_haps
        n_pad_snps = self.num_snps - n_snps

        # start and end tokens
        bos_vec = np.full((n_haps, 1), self.BOS_TOKEN)
        eos_vec = np.full((n_haps, 1), self.EOS_TOKEN)
        zeros_vec = np.zeros((n_haps, 1))

        haps = np.hstack([bos_vec, sample[:, :n_snps, 0], eos_vec]).astype(np.int8)
        
        dists = np.hstack([zeros_vec, sample[:, :n_snps, 1], zeros_vec])
        # max_dist = np.max(np.abs(dists))
        # print(f"Distances max element: {max_dist}")
        # print(f"Distances shape: {dists.shape}, dtype: {dists.dtype}")
        dists = dists.astype(np.float16)

        if n_pad_snps > 0:
            pad_vec = np.full((n_haps, n_pad_snps), self.PAD_TOKEN)
            zeros_pad_vec = np.zeros((n_haps, n_pad_snps))
            haps = np.hstack([haps, pad_vec])
            dists = np.hstack([dists, zeros_pad_vec])

        if n_pad_haps > 0:
            pad_vec = np.full((n_pad_haps, self.num_snps + 2), self.PAD_TOKEN)
            haps = np.vstack([haps, pad_vec])

        return haps, dists[0]


def make_features(
    tokenizer: Tokenizer,
    label_dtype: str | None = None,
    label_resolution: str = None,
    include_pop: bool = False,
    include_pos: bool = False,
    include_snp_pos: bool = False,
    include_major_minor: bool = False,
    include_s: bool = False,
):
    features = {
        "input_ids": Array2D((tokenizer.max_haps, tokenizer.num_snps + 2), "int8"),
        "distances": List(Value("float16")),
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
    if include_s:
        features["s"] = Value("float16")
    if include_major_minor:
        features["major_allele_flipped"] = List(Value("bool"))

    if label_dtype is not None:
        if label_resolution == "window":
            features["label"] = Value(label_dtype)
        elif label_resolution == "snp":
            features["label"] = List(Value(label_dtype))
        elif label_resolution == "snphap":
            features["label"] = Array2D(
                (tokenizer.max_haps, tokenizer.num_snps + 2),
                label_dtype,
            )
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


def hdf5_to_dataset(
    filepath: str,
    tokenizer: Tokenizer,
    window_jump: int,
    window_size: int,
    chrom=None,
    bed_file=None,
) -> Dataset:
    # we're not shuffling: get windows in order from all chroms
    # or user specified chroms
    if chrom is not None:
        chroms = [chrom]
    else:
        chroms = list(range(1, 23))

    def gen():
        # use pg gan iterator to get region
        it = RealDataRandomIterator(filename=filepath, bed_file=bed_file)
        for chrom in chroms:
            bound = it._chrom_bounds(chrom)
            tqdm.write(f"{chrom} | {bound[0]} - {bound[1]}")
            pos, i = 0, 0
            while pos < bound[1]:
                pos = it.find(i, chrom)
                i = i + window_jump

                region = it.real_region(
                    tokenizer.num_snps,
                    region_len=True,
                    region_len_size=window_size,
                    start_idx=pos,
                    return_pos=True,
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
                    # "start_pos": s,
                    # "end_pos": e,
                    # "chrom": c,
                    # "pop": filepath,
                }

    # features = make_features(tokenizer, include_pos=True, include_pop=True)
    features = make_features(tokenizer, include_pos=False, include_pop=False)
    return Dataset.from_generator(gen, features=features)


def trees_to_dataset(
    filepath: str, tokenizer: Tokenizer, window_jump: int, window_size: int
) -> Dataset:
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
            end_bp = start_bp + window_size
            start_idx = int(np.searchsorted(cum_pos, start_bp, side="left"))
            end_idx = int(np.searchsorted(cum_pos, end_bp, side="left"))

            m = gt_matrix[:, start_idx:end_idx]
            d = dist_vec[start_idx:end_idx].copy()
            p = positions[start_idx:end_idx]
            d[0] = 0

            dist = d[None, :].repeat(m.shape[0], axis=0)
            region = np.dstack([m, dist])
            region, distances = tokenizer(region)

            yield {
                "input_ids": region,
                "distances": distances,
                "chrom": 0,
                "positions": p,
                "pop": filepath,
            }

            start_bp = start_bp + window_jump

    features = make_features(include_snp_pos=True, include_pop=True)
    return Dataset.from_generator(gen, features=features)



def parse_files_imputation(ref_file: str, tgt_file: str, tokenizer: Tokenizer, bed_file = None) -> Dataset:
    samples_list = []
    it = RealDataRandomIterator(ref_file, bed_file)
    it2 = RealDataRandomIterator(tgt_file, bed_file)
    n_snps = it.num_snps
    n_snps_tgt = it2.num_snps
    n_haps_ref = it.num_samples
    n_haps = it.num_samples + it2.num_samples

    region = positions = None

    snp_ref = snp_tgt = 0
    while snp_ref < n_snps and snp_tgt < n_snps_tgt:
        cur_idx = snp_ref % tokenizer.num_snps
        if cur_idx == 0:
            if snp_tgt != 0:
                # save if not first
                # region = shuffle(region, n)
                dist_vec = [0] + [
                    (positions[j + 1] - positions[j])
                    for j in range(len(positions) - 1)
                ]

                region, flipped = process_gt_dist(
                    region, dist_vec, tokenizer.num_snps, region_len=False, ret_major_minor=True
                )

                region, distances = tokenizer(region)

                samples_list.append(
                    {
                        "input_ids": region,
                        "distances": distances,
                        "positions": positions,
                        "chrom": 0, # TODO
                        "major_allele_flipped": flipped,
                    }
                )

            region = np.zeros((tokenizer.num_snps, n_haps))
            positions = np.zeros((tokenizer.num_snps,))
        pos_ref = it.pos_all[snp_ref]
        pos_tgt = it2.pos_all[snp_tgt]

        while pos_ref < pos_tgt and snp_ref < n_snps - 1:
            # mask the tgt sample at this pos
            region[cur_idx, n_haps_ref:] = 4
            region[cur_idx, :n_haps_ref] = it.haps_all[snp_ref, :]
            positions[cur_idx] = it.pos_all[snp_ref]

            snp_ref += 1
            cur_idx = snp_ref % tokenizer.num_snps
            pos_ref = it.pos_all[snp_ref]

        # Check bounds before accessing
        if snp_ref >= n_snps or snp_tgt >= n_snps_tgt:
            break

        region[cur_idx, n_haps_ref:] = it2.haps_all[snp_tgt, :]
        region[cur_idx, :n_haps_ref] = it.haps_all[snp_ref, :]
        positions[cur_idx] = it.pos_all[snp_ref]

        snp_ref += 1
        snp_tgt += 1

    # Handle the last region if it wasn't saved
    if region is not None and positions is not None:
        dist_vec = [0] + [
            (positions[j + 1] - positions[j])
            for j in range(len(positions) - 1)
        ]
        region, flipped = process_gt_dist(
            region, dist_vec, tokenizer.num_snps, region_len=False, ret_major_minor=True
        )
        region, distances = tokenizer(region)
        samples_list.append(
            {
                "input_ids": region,
                "distances": distances,
                "positions": positions,
                "chrom": 0, # TODO
                "major_allele_flipped": flipped,
            }
        )

    features = make_features(tokenizer, include_snp_pos=True, include_major_minor=True)
    # Save tokenized data
    dataset = Dataset.from_list(samples_list, features=features)
    return dataset

def parse_file(filepath, args) -> Dataset:
    if os.path.isdir(filepath):
        return None
    ext = os.path.splitext(filepath)[1]
    if ext == ".vcf":
        # convert to h5
        newfile = filepath.replace(".vcf", ".h5")
        allel.vcf_to_hdf5(filepath, newfile, fields=["CHROM", "GT", "POS"])
        ext = ".h5"
        filepath = newfile

    tokenizer = Tokenizer(max_haps=args.max_haps, num_snps=args.num_snps)
    if ext == ".h5":
        return hdf5_to_dataset(filepath, tokenizer, args.window_jump, args.window_size, chrom=args.chrom, bed_file=args.bed_file)
    elif ext == ".trees":
        return trees_to_dataset(filepath, tokenizer, args.window_jump, args.window_size)
    else:
        # non-.vcf, .h5 filetype
        return None


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
        "--chrom",
        type=int,
        default=None,
        help="Chromosome to process. Default is all human autosomes (1-22).",
    )
    parser.add_argument("--num_snps", type=int, default=512, help="Number of SNPs.")
    parser.add_argument("--max_haps", type=int, default=256, help="Maximum haplotypes.")
    parser.add_argument(
        "--window_jump", type=int, default=50000, help="Distance between windows."
    )
    parser.add_argument(
        "--window_size", type=int, default=50000, help="Window size in base pairs."
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # directory
        files = os.listdir(args.input)
        datasets = []
        for file in files:
            filepath = os.path.join(args.input, file)
            dataset = parse_file(filepath, args)
            if dataset is None:
                print(f"Skipping unsupported file: {filepath}")
                continue
            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)

    else:
        # single file
        dataset = parse_file(args.input, args)

    dataset.save_to_disk(args.output)
