"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.

From pg_gan.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import h5py
import numpy as np
from . import util


class Region:
    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = str(chrom)
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.region_len = self.end_pos - self.start_pos  # L

    def __str__(self):
        s = str(self.chrom) + ":" + str(self.start_pos) + "-" + str(self.end_pos)
        return s

    def inside_mask(self, mask_dict, frac_callable=0.5):
        if mask_dict is None:
            return True

        if self.chrom not in mask_dict:
            return False

        mask_lst = mask_dict[self.chrom]  # restrict to this chrom
        region_start_idx, start_inside = binary_search(self.start_pos, mask_lst)
        region_end_idx, end_inside = binary_search(self.end_pos, mask_lst)

        # same region index
        if region_start_idx == region_end_idx:
            if start_inside and end_inside:  # both inside
                return True
            elif (not start_inside) and (not end_inside):  # both outside
                return False
            elif start_inside:
                part_inside = mask_lst[region_start_idx][1] - self.start_pos
            else:
                part_inside = self.end_pos - mask_lst[region_start_idx][0]
            return part_inside / self.region_len >= frac_callable

        # different region index
        part_inside = 0
        # conservatively add at first
        for region_idx in range(region_start_idx + 1, region_end_idx):
            part_inside += mask_lst[region_idx][1] - mask_lst[region_idx][0]

        # add on first if inside
        if start_inside:
            part_inside += mask_lst[region_start_idx][1] - self.start_pos
        elif self.start_pos >= mask_lst[region_start_idx][1]:
            # start after closest region, don't add anything
            pass
        else:
            part_inside += mask_lst[region_start_idx][1] - mask_lst[region_start_idx][0]

        # add on last if inside
        if end_inside:
            part_inside += self.end_pos - mask_lst[region_end_idx][0]
        elif self.end_pos <= mask_lst[region_end_idx][0]:
            # end before closest region, don't add anything
            pass
        else:
            part_inside += mask_lst[region_end_idx][1] - mask_lst[region_end_idx][0]

        return part_inside / self.region_len >= frac_callable


def read_mask(filename):
    """Read from bed file"""

    mask_dict = {}
    f = open(filename, "r")

    for line in f:
        tokens = line.split()
        chrom_str = tokens[0][3:]
        if chrom_str != "X" and chrom_str != "Y":
            begin = int(tokens[1])
            end = int(tokens[2])

            if chrom_str in mask_dict:
                mask_dict[chrom_str].append([begin, end])
            else:
                mask_dict[chrom_str] = [[begin, end]]

    f.close()
    return mask_dict


def binary_search(q, lst):
    low = 0
    high = len(lst) - 1

    while low <= high:
        mid = (low + high) // 2
        if lst[mid][0] <= q <= lst[mid][1]:  # inside region
            return mid, True
        elif q < lst[mid][0]:
            high = mid - 1
        else:
            low = mid + 1

    return mid, False  # something close


class RealDataRandomIterator:
    def __init__(self, filename, bed_file=None):
        callset = h5py.File(filename, mode="r")
        # output: ['GT'] ['CHROM', 'POS']

        raw = callset["calldata/GT"]
        newshape = (raw.shape[0], -1)
        self.haps_all = np.reshape(raw, newshape)
        self.pos_all = callset["variants/POS"]
        # same length as pos_all, noting chrom for each variant (sorted)
        self.chrom_all = callset["variants/CHROM"]
        self.num_samples = self.haps_all.shape[1]

        self.num_snps = len(self.pos_all)  # total for all chroms

        # mask
        self.mask_dict = read_mask(bed_file) if bed_file is not None else None

        # cache for chrom bounds to speed repeated queries
        self._chrom_bounds_cache = {}

    # LZ: new helpers for memory-friendly binary searches
    def _chrom_value(self, idx):
        """Return parsed integer chromosome at global index idx."""
        v = self.chrom_all[idx]
        return int(v)

    def _chrom_bounds(self, chrom):
        """Return (start, end) indices for chrom (end exclusive). Uses cache."""
        if chrom in self._chrom_bounds_cache:
            return self._chrom_bounds_cache[chrom]
        n = self.num_snps
        # find left boundary (first index with chrom >= target)
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._chrom_value(mid) < chrom:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        # early exit if chrom not present
        if left >= n or self._chrom_value(left) != chrom:
            self._chrom_bounds_cache[chrom] = (-1, -1)
            return -1, -1
        # find right boundary (first index with chrom > target)
        lo, hi = left, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._chrom_value(mid) <= chrom:
                lo = mid + 1
            else:
                hi = mid
        right = lo
        self._chrom_bounds_cache[chrom] = (left, right)
        return left, right

    def _search_pos_within(self, chrom_left, chrom_right, pos):
        """Binary search last index within [chrom_left, chrom_right) whose position <= pos.
        Returns global index (>= chrom_left) or chrom_left if pos precedes first SNP.
        """
        lo, hi = chrom_left, chrom_right  # invariant: answer in [chrom_left-1, hi-1]
        # We want last idx with pos_all[idx] <= pos. Use upper_bound style.
        while lo < hi:
            mid = (lo + hi) // 2
            mid_pos = int(self.pos_all[mid])
            if mid_pos <= pos:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < chrom_left:
            return chrom_left  # earliest index in chrom (pos before first SNP)
        return idx

    def find(self, pos, chrom):
        """Return the global SNP index of the first SNP at or before pos on chrom.
        Memory-friendly manual binary searches over h5 datasets (no numpy.searchsorted).
        If chrom not present, returns -1.
        """
        # normalize chrom type in case caller passed string
        left, right = self._chrom_bounds(chrom)
        if left == -1:
            return -1
        idx = self._search_pos_within(left, right, pos)
        return idx

    def find_end(self, start_idx, region_len_size):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chrom = util.parse_chrom(self.chrom_all[start_idx])
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < region_len_size:
            if len(self.pos_all) <= i + 1:
                print("not enough on chrom", chrom)
                return -1  # not enough on last chrom

            next_pos = self.pos_all[i + 1]
            if util.parse_chrom(self.chrom_all[i + 1]) == chrom:
                diff = next_pos - curr_pos
                ln += diff
            else:
                return -1  # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i  # exclusive

    def real_region(
        self,
        num_snps,
        region_len=False,
        region_len_size=None,
        start_idx=None,
        return_pos=False,
        return_all_pos=False,
        frac_callable=0.5,
    ):
        """
        Get a real region from the data.
        """

        if region_len:
            end_idx = self.find_end(start_idx, region_len_size)
            if end_idx == -1:
                return "end_chrom"

        else:
            end_idx = start_idx + num_snps  # exclusive
            if end_idx >= self.num_snps:
                return None

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx - 1]  # inclusive here

        if start_chrom != end_chrom:
            return None

        hap_data = self.haps_all[start_idx:end_idx, :]
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx]
        positions = self.pos_all[start_idx:end_idx]

        chrom = util.parse_chrom(start_chrom)
        region = Region(chrom, start_base, end_base)
        result = region.inside_mask(self.mask_dict, frac_callable=frac_callable)

        # if we do have an accessible region
        if result:
            # if region_len, then positions_S is actually positions_len
            dist_vec = [0] + [
                (positions[j + 1] - positions[j]) for j in range(len(positions) - 1)
            ]

            after = util.process_gt_dist(
                hap_data,
                dist_vec,
                num_snps,
                region_len=region_len,
            )
            if return_all_pos:
                return after, positions, chrom
            elif return_pos:
                return after, start_base, end_base, chrom
            return after  # , [chrom, start_base, end_base]

        return None
