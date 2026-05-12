"""
Utilities for generating Huggingface datasets for haplotype windows.
"""

import argparse
import os

import allel
import numpy as np
import tskit
from datasets import Array2D, Dataset, Features, List, Value, concatenate_datasets
from tqdm import tqdm

from .real_data_random import RealDataRandomIterator
from .util import major_minor, process_gt_dist


def get_pos_and_dist_vec(ts, snps_total, mask=None):
    positions = [round(variant.site.position) for variant in ts.variants()]
    positions = [pos for i, pos in enumerate(positions) if mask is None or mask[i]]
    assert len(positions) == snps_total

    dist_vec = [0] + [(positions[j + 1] - positions[j]) for j in range(snps_total - 1)]
    return positions, np.array(dist_vec)


class Tokenizer:
    BOS_TOKEN = 2
    EOS_TOKEN = 3
    MASK_TOKEN = 4
    PAD_TOKEN = 5

    def __init__(self, max_haps: int, num_snps: int, major_minor_flip=True):
        self.max_haps = max_haps
        self.num_snps = num_snps
        self.major_minor_flip = major_minor_flip

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return self.tokenizer(sample)

    def get_config(self) -> dict:
        return {
            "max_haps": self.max_haps,
            "num_snps": self.num_snps,
            "major_minor_flip": self.major_minor_flip,
        }

    def tokenizer(self, sample: np.ndarray) -> np.ndarray:
        if np.any(sample[:, :, 0] == 2):
            raise ValueError("Genotype matrix has unexpected value 2.")

        # ensure major/minor
        if self.major_minor_flip:
            sample[:, :, 0], _ = major_minor(sample[:, :, 0])

        # padding
        n_haps = min(sample.shape[0], self.max_haps)
        n_snps = min(sample.shape[1], self.num_snps)
        n_pad_haps = self.max_haps - n_haps
        n_pad_snps = self.num_snps - n_snps

        # truncation if needed. take window centered around middle if too many
        if sample.shape[1] > self.num_snps:
            mid = sample.shape[1] // 2
            snp_start = max(0, mid - self.num_snps // 2)
            snp_end = snp_start + self.num_snps
        else:
            snp_start = 0
            snp_end = sample.shape[1]

        # start and end tokens
        bos_vec = np.full((n_haps, 1), self.BOS_TOKEN)
        eos_vec = np.full((n_haps, 1), self.EOS_TOKEN)
        zeros_vec = np.zeros((n_haps, 1))

        haps = np.hstack(
            [bos_vec, sample[:n_haps, snp_start:snp_end, 0], eos_vec]
        ).astype(np.int8)

        dists = np.hstack([zeros_vec, sample[:n_haps, snp_start:snp_end, 1], zeros_vec])
        # max_dist = np.max(np.abs(dists))
        # print(f"Distances max element: {max_dist}")
        # print(f"Distances shape: {dists.shape}, dtype: {dists.dtype}")
        # dists = dists.astype(np.float16)

        if n_pad_snps > 0:
            pad_vec = np.full((n_haps, n_pad_snps), self.PAD_TOKEN)
            zeros_pad_vec = np.zeros((n_haps, n_pad_snps))
            haps = np.hstack([haps, pad_vec])
            dists = np.hstack([dists, zeros_pad_vec])

        if n_pad_haps > 0:
            pad_vec = np.full((n_pad_haps, self.num_snps + 2), self.PAD_TOKEN)
            haps = np.vstack([haps, pad_vec])

        # any nans?
        if np.isnan(dists).any():
            raise ValueError("Distances contain NaN values.")

        if np.isnan(haps).any():
            raise ValueError("Haplotypes contain NaN values.")

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
    include_shoulder: bool = False,
    extra_features: dict[str, str] = None,  # <-- new parameter
):
    features = {
        "input_ids": Array2D((tokenizer.max_haps, tokenizer.num_snps + 2), "int8"),
        "distances": List(Value("int32")),
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
    if include_shoulder:
        features["shoulder"] = Value("int8")
    if include_major_minor:
        features["major_allele_flipped"] = List(Value("bool"))

    if extra_features is not None:
        for k, v in extra_features.items():
            features[k] = Value(v)

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
    frac_callable=0.5,
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
                    frac_callable=frac_callable,
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
                    # "pop": filepath,
                }

    features = make_features(tokenizer, include_pos=True, include_pop=False)
    # features = make_features(tokenizer, include_pos=False, include_pop=False)
    return Dataset.from_generator(gen, features=features)


def trees_to_dataset(
    filepath: str, tokenizer: Tokenizer, window_jump: int, window_size: int, pop=None
) -> Dataset:
    def gen():
        # process tree
        ts = tskit.load(filepath)
        pops = ts.populations()

        if len(pops) > 1 and pop is not None:
            pop_ids = [
                p.id
                for p in pops
                if p.metadata is not None and p.metadata["name"] == pop
            ]
            if len(pop_ids) == 0:
                raise ValueError(f"Population {pop} not found in tree sequence.")
            pop_id = pop_ids[0]
            samples = ts.samples(population=pop_id)
            tqdm.write(f"Found population {pop} with {len(samples)} samples.")
            if len(samples) > 256:
                samples = samples[:256]
            ts = ts.simplify(samples=samples)
            tqdm.write(f"Filtered to population {pop} with {ts.num_samples} samples.")

        gt_matrix = ts.genotype_matrix()
        is_biallelic = [
            sum(gt_matrix[i]) == list(gt_matrix[i]).count(1)
            for i in range(len(gt_matrix))
        ]
        gt_matrix = gt_matrix[is_biallelic]
        num_snps = gt_matrix.shape[0]
        positions, dist_vec = get_pos_and_dist_vec(ts, num_snps, is_biallelic)

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

    features = make_features(tokenizer, include_snp_pos=True, include_pop=True)
    return Dataset.from_generator(gen, features=features)


def ms_to_dataset(
    filepath: str,
    tokenizer: Tokenizer,
    label=None,
    label_dtype=None,
    extra_vars=None,
    extra_vars_dtypes=None,
) -> Dataset:
    # parse ms output file and convert to dataset
    def gen():
        ms = ""
        with open(filepath, "r") as f:
            ms = f.read().splitlines()

        # first 3 lines are comments
        cmd = ms[0]
        n_haps = int(cmd.split()[1])
        n_sims = int(cmd.split()[2])
        length = int(cmd.split()[3])

        sim_starts = [i for i, line in enumerate(ms) if line.startswith("//")]
        assert len(sim_starts) == n_sims, (
            f"Expected {n_sims} simulations but found {len(sim_starts)} in file."
        )

        for i, start_idx in enumerate(sim_starts):
            # each sample starts with //, followed by 2 lines of metadata, then the haplotype matrix
            end_idx = start_idx + n_haps + 2 + 1

            matrix = []
            positions = []
            for j in range(start_idx, end_idx):
                line = ms[j]
                if line.startswith("//"):
                    # new sample
                    continue
                elif line.startswith("segsites"):
                    n_snps = int(line.split()[1])
                elif line.startswith("positions"):
                    positions = [float(x) for x in line.split()[1:]]
                else:
                    # haplotype matrix
                    haps = np.array([int(x) for x in line.strip()])
                    matrix.append(haps)

            matrix = np.array(matrix, dtype=np.int32)
            distances = (
                np.array(
                    [0]
                    + [
                        positions[j + 1] - positions[j]
                        for j in range(len(positions) - 1)
                    ]
                )
                * length
            ).astype(np.int32)
            dist = distances[None, :].repeat(matrix.shape[0], axis=0)
            region = np.dstack([matrix, dist])

            # use tokenizer to convert to input_ids and distances
            region, distances = tokenizer(region)

            out = {
                "input_ids": region,
                "distances": distances,
            }
            if label is not None:
                out["label"] = label
            if extra_vars is not None:
                for var_name, var_value in extra_vars.items():
                    out[var_name] = var_value

            yield out

    features = make_features(
        tokenizer,
        label_dtype=label_dtype,
        label_resolution="window",
        extra_features=extra_vars_dtypes,
    )

    return Dataset.from_generator(gen, features=features)


def parse_files_imputation(
    ref_file: str, tgt_file: str, tokenizer: Tokenizer, bed_file=None
) -> Dataset:
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
                    (positions[j + 1] - positions[j]) for j in range(len(positions) - 1)
                ]

                region, flipped = process_gt_dist(
                    region,
                    dist_vec,
                    tokenizer.num_snps,
                    region_len=False,
                    ret_major_minor=True,
                )

                region, distances = tokenizer(region)

                samples_list.append(
                    {
                        "input_ids": region,
                        "distances": distances,
                        "positions": positions,
                        "chrom": 0,  # TODO
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
            (positions[j + 1] - positions[j]) for j in range(len(positions) - 1)
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
                "chrom": 0,  # TODO
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
    if ext == ".vcf" or ext == ".gz":
        # convert to h5
        newfile = filepath.replace(".gz", "").replace(".vcf", ".h5")
        allel.vcf_to_hdf5(
            filepath, newfile, fields=["CHROM", "GT", "POS"], overwrite=True
        )
        ext = ".h5"
        filepath = newfile

    tokenizer = Tokenizer(max_haps=args.max_haps, num_snps=args.num_snps)
    if ext == ".h5":
        return hdf5_to_dataset(
            filepath,
            tokenizer,
            args.window_jump,
            args.window_size,
            chrom=args.chrom,
            bed_file=args.bed_file,
            frac_callable=args.frac_callable,
        )
    elif ext == ".trees":
        return trees_to_dataset(
            filepath, tokenizer, args.window_jump, args.window_size, pop=args.pop
        )
    elif ext == ".msout":
        return ms_to_dataset(filepath, tokenizer)
    else:
        # non-.vcf, .h5 filetype
        return None


def main():
    parser = argparse.ArgumentParser(description="Process an input file/directory.")
    parser.add_argument(
        "input",
        help="Input file or directory. Supported filetypes: "
        "- vcf (converted to hdf5)"
        "- hdf5"
        "- txt (in ms format)"
        "- .trees (output from stdpopsim)"
        "- .msout (output from ms)"
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
    parser.add_argument(
        "--frac_callable",
        type=float,
        default=0.5,
        help="Minimum fraction of callable sites in a window.",
    )
    parser.add_argument(
        "--pop",
        type=str,
        default=None,
        help="Only used if tree sequence contains multiple populations. Will filter to this population.",
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


if __name__ == "__main__":
    main()
