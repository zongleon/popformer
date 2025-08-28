"""
Utilities for a dataset.
"""

import argparse
import os

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from pg_gan import global_vars
from pg_gan.generator import Generator
from pg_gan.global_vars import DEFAULT_SEED, NUM_SNPS
from pg_gan.real_data_random import RealDataRandomIterator
from pg_gan.ss_helpers import parse_output
from pg_gan.util import parse_args, process_opts

VERSION=2
OUTFILE_PATH = "outfiles/{pop}/{pop}_{seed}_{model}.out"
GENOME_PATH = (
    "/bigdata/smathieson/1000g-share/HDF5/{pop}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
)
BED_PATH = "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed"

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


def get_iterator(pop: str, seed=None) -> RealDataRandomIterator:
    h5_file = GENOME_PATH.format(pop=pop)
    bed_file = BED_PATH
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

def get_data(
    pop: str, n_samples: int, seed=None
) -> np.ndarray:
    """
    Get a dataset of real and simulated data.

    Returns a tuple of (samples, labels, sources):
        Samples is a numpy array of shape (n_samples, num_snps, 2).
    """
    iterator = get_iterator(pop=pop, seed=seed)

    # get n_samples from each iterator
    total_n = n_samples
    samples = np.empty((total_n, iterator.num_samples, NUM_SNPS, 2),
                       dtype=np.float32)
    print("Sampling from iterator:", pop)
    for i in tqdm(range(n_samples)):
        sample = iterator.real_region(neg1=False, region_len=False)
        samples[i] = sample

    return samples


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


def hapiter(hap_sample: np.ndarray):
    for hap in hap_sample:
        yield hap


def compute_token_distances_simple(hap: np.ndarray, distances: np.ndarray):
    """
    Compute distances from middle of one token to middle of the next token.

    Args:
        hap: Array representation of haplotype
        distances: Array of distances between consecutive SNPs
        tokenizer: Trained tokenizer

    Returns:
        Tuple of encodings, distances between token middles
    """
    return (
        np.concatenate([[2], hap, [3]]),
        np.concatenate([[0], distances, [0]])
    )

if __name__ == "__main__":
    # "Usage: python dataset.py <gen | tokenize> [n_samples] pop")
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument("mode", choices=["gen",
                                         "runsimple", "runrealsim", "runsel", "runsel2",
                                         "ghist", "admix"], 
                                         help="Mode to run")
    parser.add_argument("n_samples", type=int, nargs="?", default=1000, help="Number of samples")
    args = parser.parse_args()

    mode = args.mode
    n_samples = args.n_samples

    if mode == "gen":
        for pop in ["CEU", "CHB", "YRI"]:
            samples = get_data(
                pop=pop, n_samples=n_samples, seed=0
            )
            save_data(pop, samples)
    elif mode == "runsimple":
        # Now compute tokenized data with distances
        tokenized_data = []

        def gen():
            for pop in ["CEU", "CHB", "YRI"]:
                samples = load_data(pop)
                haps = samples[..., 0].astype(np.int8)
                distances = samples[..., 1].astype(np.float32) * global_vars.L

                for hap, dist in tqdm(zip(haps, distances), total=samples.shape[0]):
                    encodings_all = []
                    for hap, distances in hapiter(hap):
                        encodings, token_distances = compute_token_distances_simple(hap, dist[0])

                        encodings_all.append(encodings)
                    # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
                    yield {
                        'input_ids': encodings_all,
                        'distances': token_distances,
                        'pop': pop,
                    }

        # Save tokenized data
        dataset = Dataset.from_generator(gen)
        dataset.save_to_disk(f"dataset{VERSION}/tokenized")

    elif mode == "runrealsim":
        tokenized_data = []

        for pop in ["CEU", "CHB", "YRI"]:
            samples = np.load(f"../disc-interpret/dataset-{pop}/X.npy")
            labels = np.load(f"../disc-interpret/dataset-{pop}/y.npy")

            subset = np.random.choice(samples.shape[0], n_samples, replace=False)
            samples = samples[subset]
            labels = labels[subset]
            haps = samples[..., 0].astype(np.int8)
            haps[haps == -1] = 0
            distances = samples[..., 1].astype(np.float32) * global_vars.L

            for hap, dist, label in tqdm(zip(haps, distances, labels), total=samples.shape[0]):
                encodings_all = []
                for hap_str, distances in popiter(hap, dist):
                    encodings, token_distances = compute_token_distances_simple(hap_str, 
                                                                                distances)
                    encodings_all.append(encodings)
                    
                # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
                tokenized_data.append({
                    'input_ids': encodings_all,
                    'distances': token_distances,
                    'label': label
                })

        # Save tokenized data
        dataset = Dataset.from_list(tokenized_data)
        dataset.save_to_disk("dataset/tokenizedft")
    elif mode == "runsel":
        tokenized_data = []

        for t, sel in zip(
            ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
            [0, 1.0, 0.1, 0.5, 0.25]
        ):
            for pop in ["CEU", "CHB", "YRI"]:
                print(f"Processing {pop} {t}")
                smp: np.ndarray = np.load(f"1000g/{pop}/matrices_{pop}_{t}.npy")[:2400]
                smp = smp.astype(np.int8)
                distances: np.ndarray = np.load(f"1000g/{pop}/distances_{pop}_{t}.npy")

                for haps, dist in zip(smp, distances):
                    encodings_all = []
                    for hap in hapiter(haps.T):
                        encodings, token_distances = compute_token_distances_simple(hap.T,
                                                                                    dist)
                        encodings_all.append(encodings)
                    # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
                    tokenized_data.append({
                        'input_ids': encodings_all,
                        'distances': token_distances,
                        'label': sel
                    })

        # Save tokenized data
        dataset = Dataset.from_list(tokenized_data)
        dataset.save_to_disk("dataset/tokenizedsel")
    elif mode == "runsel2":
        tokenized_data = []

        for t, sel in zip(
            ["neutral_3000", "sel1_600", "sel01_600", "sel05_600", "sel025_600"],
            [0, 1, 1, 1, 1]
        ):
            for pop in ["CEU", "CHB", "YRI"]:
                print(f"Processing {pop} {t}")
                smp: np.ndarray = np.load(f"1000g/{pop}/matrices_{pop}_{t}.npy")[:2400]
                smp = smp.astype(np.int8)
                distances: np.ndarray = np.load(f"1000g/{pop}/distances_{pop}_{t}.npy")

                for haps, dist in zip(smp, distances):
                    encodings_all = []
                    for hap in hapiter(haps.T):
                        encodings, token_distances = compute_token_distances_simple(hap.T,
                                                                                    dist)
                        encodings_all.append(encodings)
                            
                    # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
                    tokenized_data.append({
                        'input_ids': encodings_all,
                        'distances': token_distances,
                        'label': sel
                    })

        # Save tokenized data
        dataset = Dataset.from_list(tokenized_data)
        dataset.save_to_disk("dataset/tokenizedsel2")
    elif mode == "ghist":
        t = "multisweep.growth_bg"
        it = get_iterator_ghist(f"GHIST/GHIST_2025_{t}.21.testing_process.h5", "GHIST/raw/21.accessible.bed")
        n_snps = it.num_snps

        # every 18 snps, sample a region
        smps = []
        pos = []
        for i in tqdm(range(0, n_snps, 18)):
            region = it.real_region(neg1=False, region_len=False, start_idx=i)
            if region is not None:
                smps.append(region)
                pos.append(it.pos_all[i])

        smps = np.array(smps)
        pos = np.array(pos)
        haps = smps[..., 0].astype(np.int8)
        distances = smps[..., 1].astype(np.float32) * global_vars.L
        
        tokenized_data = []

        for haps, dist, p in zip(haps, distances, pos):
            encodings_all = []
            for hap in hapiter(haps):
                encodings, token_distances = compute_token_distances_simple(hap,
                                                                            dist[0])
                encodings_all.append(encodings)
                    
            # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
            tokenized_data.append({
                'input_ids': encodings_all,
                'distances': token_distances,
                'pos': p
            })

        ds = Dataset.from_list(tokenized_data)
        ds.save_to_disk(f"GHIST/ghist_samples_{t}")
    elif mode == "admix":
        it = get_iterator_ghist("/bigdata/smathieson/1000g-share/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5", 
                                "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed")
        n_snps = it.num_snps

        # every 18 snps, sample a region
        smps = []
        pos = []
        for i in tqdm(range(0, n_snps, 128)):
            if n_samples is not None and i / 128 > n_samples:
                break
            region = it.real_region(neg1=False, region_len=False, start_idx=i)
            if region is not None:
                smps.append(region)
                pos.append([it.chrom_all[i], it.pos_all[i]])

        smps = np.array(smps)
        pos = np.array(pos)
        haps = smps[..., 0].astype(np.int8)
        distances = smps[..., 1].astype(np.float32) * global_vars.L
        
        tokenized_data = []

        for idx, (haps, dist, p) in enumerate(zip(haps, distances, pos)):
            encodings_all = []
            for hap in hapiter(haps):
                encodings, token_distances = compute_token_distances_simple(hap,
                                                                            dist[0])
                encodings_all.append(encodings)
                    
            # save list of lists of (n_haps, n_input_ids) and (n_haps, n_distances)
            tokenized_data.append({
                'input_ids': encodings_all,
                'distances': token_distances,
                'pos': p
            })

        ds = Dataset.from_list(tokenized_data)
        ds.save_to_disk("LAI/LAI_CEU_test")