"""
Utility functions and classes (including default parameters).

Consolidated form pg_gan. --
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np


def process_gt_dist(
    gt_matrix, dist_vec, num_snps, region_len=False
):
    """
    Take in a genotype matrix and vector of inter-SNP distances. Return a 3D
    numpy array of the given n (haps) and S (SNPs) and 2 channels.
    Filter singletons at given rate if filter=True
    """

    num_SNPs = gt_matrix.shape[0]  # SNPs x n
    n = gt_matrix.shape[1]

    # double check
    if num_SNPs != len(dist_vec):
        print("gt", num_SNPs, "dist", len(dist_vec))
    assert num_SNPs == len(dist_vec)

    # used for trimming (don't trim if using the entire region)
    S = num_SNPs if region_len else num_snps

    # set up region
    region = np.zeros((n, S, 2), dtype=np.float32)

    mid = num_SNPs // 2
    half_S = S // 2
    if S % 2 == 1:  # odd
        other_half_S = half_S + 1
    else:
        other_half_S = half_S

    # enough SNPs, take middle portion
    if mid >= half_S:
        minor = major_minor(
            gt_matrix[mid - half_S : mid + other_half_S, :].transpose()
        )
        region[:, :, 0] = minor
        distances = np.vstack(
            [np.copy(dist_vec[mid - half_S : mid + other_half_S]) for k in range(n)]
        )
        region[:, :, 1] = distances

    # not enough SNPs, need to center-pad
    else:
        minor = major_minor(gt_matrix.T)
        region[:, half_S - mid : half_S - mid + num_SNPs, 0] = minor
        distances = np.vstack([np.copy(dist_vec) for k in range(n)])
        region[:, half_S - mid : half_S - mid + num_SNPs, 1] = distances

    return region  # n X SNPs X 2


def major_minor(matrix):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        if np.count_nonzero(matrix[:, j] > 0) > (n / 2):  # count the 1's
            matrix[:, j] = 1 - matrix[:, j]

    return matrix


def parse_chrom(chrom):
    if isinstance(chrom, bytes):
        return chrom.decode("utf-8")

    return chrom  # hg19 option