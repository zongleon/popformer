"""
Utilities for a dataset.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from datasets import Dataset
from transformers import RobertaTokenizerFast

from pg_gan import global_vars
from pg_gan.real_data_random import RealDataRandomIterator
from pg_gan.generator import Generator
from pg_gan.ss_helpers import parse_output
from pg_gan.util import parse_args, process_opts
from pg_gan.global_vars import DEFAULT_SEED, NUM_SNPS

OUTFILE_PATH = "outfiles/{pop}/{pop}_{seed}_{model}.out"
GENOME_PATH = (
    "/bigdata/smathieson/1000g-share/HDF5/{pop}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
)
BED_PATH = "/bigdata/smathieson/1000g-share/HDF5/20120824_strict_mask.bed"

pop = None

_OUTPUT_SAMPLES = "dataset-{pop}/samples.npz"

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


def save_data(samples: np.ndarray):
    """
    Save the dataset to npz and csv files.
    """
    d = os.path.dirname(_OUTPUT_SAMPLES.format(pop=pop))
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    np.savez_compressed(os.path.join(d, "X.npz"), X=samples)
    print(f"Saved {len(samples)} samples with {samples.shape[1]} SNPs each.")


def load_data(pop: str, dir=None) -> np.ndarray:
    """
    Load the dataset from the npz file.
    Returns a tuple of (samples, labels).
    """
    if dir:
        file = os.path.join(dir, f"dataset-{pop}", "samples.npz")
    else:
        file = _OUTPUT_SAMPLES.format(pop=pop)
    d = os.path.dirname(file)
    if not os.path.exists(d):
        raise FileNotFoundError(f"Dataset path {file} does not exist.")

    # Load samples and labels into memory
    samples = np.load(os.path.join(d, "X.npz"))["X"]

    print(f"Loaded {len(samples)} samples with {samples.shape[1]} SNPs each.")

    return samples

def hapiter(samples: np.ndarray):
    """
    Create a haplotype iterator from the samples.
    """
    samples = samples[..., 0]
    for sample in samples:
        # sample like (n_haps, n_snps)
        for hap in sample:
            yield "".join(str(x) for x in hap)

def hapiter_with_distances(hap_samples: np.ndarray, dist_samples: np.ndarray):
    """
    Create a haplotype iterator from the samples with corresponding distances.
    """
    for hap_sample, dist_sample in zip(hap_samples, dist_samples):
        # sample like (n_haps, n_snps)
        distances = dist_sample[0]
        for hap in hap_sample:
            hap_str = "".join(str(int(x)) for x in hap)
            yield hap_str, distances

def compute_token_distances(hap_str: str, distances: np.ndarray, tokenizer):
    """
    Compute distances from middle of one token to middle of the next token.
    
    Args:
        hap_str: String representation of haplotype
        distances: Array of distances between consecutive SNPs
        tokenizer: Trained tokenizer
    
    Returns:
        Tuple of encodings, distances between token middles
    """
    # Encode the haplotype to get tokens
    encoding = tokenizer(hap_str)
    tokens = encoding.input_ids
    
    if len(tokens) <= 1:
        return encoding, np.array([])
    
    # Get the span (start, end) of each token in the original sequence
    token_spans = []
    for i, token in enumerate(tokens):
        span = encoding.token_to_chars(i)
        if span is not None:
            token_spans.append(span)
    
    # Compute middle position of each token
    token_middles = []
    for start, end in token_spans:
        middle_pos = (start + end - 1) / 2  # -1 because end is exclusive
        token_middles.append(middle_pos)
    
    # Compute cumulative distances to get absolute positions
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # Interpolate to get distances at token middle positions
    token_middle_distances = np.interp(token_middles, range(len(cumulative_distances)), cumulative_distances)
    
    # Compute distances between consecutive token middles
    token_distances = np.diff(token_middle_distances)
    
    return encoding, token_distances

if __name__ == "__main__":
    # "Usage: python dataset.py <gen | tokenize> [n_samples] pop")
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument("mode", choices=["gen", "traintokenizer", "runtokenizer"], help="Mode to run")
    parser.add_argument("n_samples", type=int, nargs="?", default=1000, help="Number of samples")
    parser.add_argument("pop", help="Population identifier")
    args = parser.parse_args()

    mode = args.mode
    n_samples = args.n_samples
    pop = args.pop

    if mode == "gen":
        samples = get_data(
            pop=pop, n_samples=n_samples, seed=0
        )
        save_data(samples)
    elif mode == "traintokenizer":
        # load samples
        samples = load_data(pop=pop)
        samples = samples.astype(np.int8)

        # tokenize
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer()

        # Train tokenizer on haplotypes only
        tokenizer.train_from_iterator(hapiter(samples), 
                                      trainer=trainer, 
                                      length=samples.shape[0] * samples.shape[1])

        # Save tokenizer
        tokenizer.model.save("tokenizer")
        # tokenizer.save("tokenizer.json")
    else:
        tokenizer = RobertaTokenizerFast(vocab_file="tokenizer/vocab.json", merges_file="tokenizer/merges.txt")
        samples = load_data(pop=pop)
        haps = samples[..., 0].astype(np.long)
        distances = samples[..., 1].astype(np.float32) * global_vars.L

        # Now compute tokenized data with distances
        tokenized_data = []

        for hap_str, distances in tqdm(hapiter_with_distances(haps, distances),
                                       total=samples.shape[0] * samples.shape[1]):
            encodings, token_distances = compute_token_distances(hap_str, distances, tokenizer)
            
            tokenized_data.append({
                'token_ids': encodings.input_ids,
                # 'token_strings': encodings.tokens,
                'distances': token_distances.tolist()
            })
        # example
        print(tokenized_data[0])
        # Save tokenized data
        dataset = Dataset.from_list(tokenized_data)
        dataset.save_to_disk(f"dataset-{pop}/tokenized")
