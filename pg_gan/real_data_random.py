"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
from collections import defaultdict
import h5py
import numpy as np
from numpy.random import default_rng
import sys
import datetime

# to remove
#import math
#import tensorflow as tf
#import discriminator

# our imports
from . import global_vars
from . import util

class Region:

    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = str(chrom)
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.region_len = self.end_pos - self.start_pos # L

    def __str__(self):
        s = str(self.chrom) + ":" + str(self.start_pos) + "-" +str(self.end_pos)
        return s

    def inside_mask(self, mask_dict, frac_callable = 0.5):
        if mask_dict is None:
            return True

        mask_lst = mask_dict[self.chrom] # restrict to this chrom
        region_start_idx, start_inside = binary_search(self.start_pos, mask_lst)
        region_end_idx, end_inside = binary_search(self.end_pos, mask_lst)

        # same region index
        if region_start_idx == region_end_idx:
            if start_inside and end_inside: # both inside
                return True
            elif (not start_inside) and (not end_inside): # both outside
                return False
            elif start_inside:
                part_inside = mask_lst[region_start_idx][1] - self.start_pos
            else:
                part_inside = self.end_pos - mask_lst[region_start_idx][0]
            return part_inside/self.region_len >= frac_callable

        # different region index
        part_inside = 0
        # conservatively add at first
        for region_idx in range(region_start_idx+1, region_end_idx):
            part_inside += (mask_lst[region_idx][1] - mask_lst[region_idx][0])

        # add on first if inside
        if start_inside:
            part_inside += (mask_lst[region_start_idx][1] - self.start_pos)
        elif self.start_pos >= mask_lst[region_start_idx][1]:
            # start after closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_start_idx][1] -
                mask_lst[region_start_idx][0])

        # add on last if inside
        if end_inside:
            part_inside += (self.end_pos - mask_lst[region_end_idx][0])
        elif self.end_pos <= mask_lst[region_end_idx][0]:
            # end before closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_end_idx][1] -
                mask_lst[region_end_idx][0])

        return part_inside/self.region_len >= frac_callable

def read_mask(filename):
    """Read from bed file"""

    mask_dict = {}
    f = open(filename,'r')

    for line in f:
        tokens = line.split()
        chrom_str = tokens[0][3:]
        if chrom_str != 'X' and chrom_str != 'Y':
            begin = int(tokens[1])
            end = int(tokens[2])

            if chrom_str in mask_dict:
                mask_dict[chrom_str].append([begin,end])
            else:
                mask_dict[chrom_str] = [[begin,end]]

    f.close()
    return mask_dict

def binary_search(q, lst):
    low = 0
    high = len(lst)-1

    while low <= high:

        mid = (low+high)//2
        if lst[mid][0] <= q <= lst[mid][1]: # inside region
            return mid, True
        elif q < lst[mid][0]:
            high = mid-1
        else:
            low = mid+1

    return mid, False # something close

class RealDataRandomIterator:

    def __init__(self, filename, seed, bed_file=None, chrom_starts=False):
        callset = h5py.File(filename, mode='r')
        print(list(callset.keys()))
        # output: ['GT'] ['CHROM', 'POS']
        print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

        raw = callset['calldata/GT']
        print("raw", raw.shape)
        newshape = (raw.shape[0], -1)
        self.haps_all = np.reshape(raw, newshape)
        self.pos_all = callset['variants/POS']
        # same length as pos_all, noting chrom for each variant (sorted)
        self.chrom_all = callset['variants/CHROM']
        print("after haps", self.haps_all.shape)
        self.num_samples = self.haps_all.shape[1]

        '''print(self.pos_all.shape)
        print(self.pos_all.chunks)
        print(self.chrom_all.shape)
        print(self.chrom_all.chunks)'''
        self.num_snps = len(self.pos_all) # total for all chroms

        # mask
        self.mask_dict = read_mask(bed_file) if bed_file is not None else None
        print(self.mask_dict)

        self.rng = default_rng(seed)

        # useful for fastsimcoal and msmc
        if chrom_starts:
            self.chrom_counts = defaultdict(int)
            for x in list(self.chrom_all):
                self.chrom_counts[int(x)] += 1
            print(self.chrom_counts)

    def find_end(self, start_idx):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chrom = global_vars.parse_chrom(self.chrom_all[start_idx])
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < global_vars.L:

            if len(self.pos_all) <= i+1:
                print("not enough on chrom", chrom)
                return -1 # not enough on last chrom

            next_pos = self.pos_all[i+1]
            if global_vars.parse_chrom(self.chrom_all[i+1]) == chrom:
                diff = next_pos - curr_pos
                ln += diff
            else:
                print("not enough on chrom", chrom)
                return -1 # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i # exclusive

    def real_region(self, neg1, region_len, start_idx=None):
        # inclusive
        recursive = True
        if start_idx is None:
            start_idx = self.rng.integers(0, self.num_snps - global_vars.NUM_SNPS)
            recursive = False
        #print('start idx', start_idx)

        if region_len:
            end_idx = self.find_end(start_idx)
            if end_idx == -1:
                if recursive:
                    return self.real_region(neg1, region_len, start_idx=start_idx) # try again
                else:
                    return None

        else:
            end_idx = start_idx + global_vars.NUM_SNPS # exclusive
            if end_idx >= self.num_snps:
                if recursive:
                    return self.real_region(neg1, region_len) # try again
                else:
                    return None

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx-1] # inclusive here

        if start_chrom != end_chrom:
            #print("bad chrom", start_chrom, end_chrom)
            if recursive:
                return self.real_region(neg1, region_len) # try again
            else:
                return None

        hap_data = self.haps_all[start_idx:end_idx, :]
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx]
        positions = self.pos_all[start_idx:end_idx]

        chrom = global_vars.parse_chrom(start_chrom)
        region = Region(chrom, start_base, end_base)
        result = region.inside_mask(self.mask_dict)

        # if we do have an accessible region
        if result:
            # if region_len, then positions_S is actually positions_len
            dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L
                for j in range(len(positions)-1)]

            after = util.process_gt_dist(hap_data, dist_vec,
                region_len=region_len, real=True, neg1=neg1)
            return after #, [chrom, start_base, end_base]

        # try again if not in accessible region
        if recursive:
            return self.real_region(neg1, region_len)
        return None

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
        region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        if not region_len:
            regions = np.zeros((batch_size, self.num_samples,
                global_vars.NUM_SNPS, 2), dtype=np.float32)
            #region_info = []

            for i in range(batch_size):
                regions[i] = self.real_region(neg1, region_len)
                #region_info.append(info)

        else:
            regions = []
            for i in range(batch_size):
                regions.append(self.real_region(neg1, region_len))

        return regions #, region_info

    def real_chrom(self, chrom, samples):
        """Mostly used for msmc - gather all data for a given chrom int"""
        start_idx = 0
        for i in range(1, chrom):
            start_idx += self.chrom_counts[i]
        end_idx = start_idx + self.chrom_counts[chrom]
        print(chrom, start_idx, end_idx)
        positions = self.pos_all[start_idx:end_idx]

        assert len(samples) == 2 # two populations
        n = self.haps_all.shape[1]
        half = n//2
        pop1_data = self.haps_all[start_idx:end_idx, 0:samples[0]]
        pop2_data = self.haps_all[start_idx:end_idx, half:half+samples[1]]
        hap_data = np.concatenate((pop1_data, pop2_data), axis=1)
        assert len(hap_data) == len(positions)

        return hap_data.transpose(), positions

# simoid
#def get_prob(x):
#    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    # testing

    # test file
    filename = sys.argv[1]
    bed_file = sys.argv[2]
    iterator = RealDataRandomIterator(filename, global_vars.DEFAULT_SEED,
                                      bed_file, chrom_starts=True)
    #out_file = open(sys.argv[4], 'w')

    disc = tf.saved_model.load(sys.argv[3])
    disc_recon = discriminator.OnePopModel(iterator.num_samples, global_vars.DEFAULT_SEED,
                                           saved_model=disc)

    start_time = datetime.datetime.now()
    #for i in range(100):
    #    region = iterator.real_region(True, False)
    #    #pred = disc(region)
    #    print("logit", pred)

    for i in range(1000):
        num_batch = 300
        regions, region_info = iterator.real_batch(batch_size=num_batch)
        logits = disc_recon(regions, training=False)
        probs = [get_prob(x) for x in logits]
        #print(probs)
        #print('min', min(probs), 'max', max(probs))
        for j in range(num_batch):
            chrom = str(region_info[j][0])
            start = str(region_info[j][1])
            end = str(region_info[j][2])
            #out_file.write("\t".join([chrom, start, end, str(probs[j])]) + "\n")
            print(chrom, start, end, probs[j])
            print(regions[j])
            input('enter')
    #out_file.close()

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print("time s:ms", elapsed.seconds,":",elapsed.microseconds)

    # test find_end
    '''for i in range(10):
        start_idx = iterator.rng.integers(0, iterator.num_snps - \
            global_vars.NUM_SNPS)
        iterator.find_end(start_idx)'''
