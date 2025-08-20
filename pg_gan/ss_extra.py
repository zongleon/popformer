# based on code by Ryan M. Cecil:
# https://github.com/ryanmcecil/popgen_ml_sweep_detection/blob/master/models/popgen_summary_statistics.py


import allel
import numpy as np
import time
from typing import List
from . import global_vars

EXTRA_STATS = ['ihs_maxabs', 'tajima_d', 'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2_h1']

def compute_extra_stats(matrix, benchmark=False):
    """Compute extra statistics with optional timing"""
    if benchmark:
        timings = {}
    
    # convert our data into their format
    data = prep_our_data(matrix)

    # compute desired statistics
    if benchmark:
        ihs_maxabs, timings['ihs_maxabs'] = predict_ihs_max(data, benchmark=True)
        tajima_d, timings['tajima_d'] = predict_td(data, benchmark=True)
        garud_h1, garud_h12, garud_h123, garud_h2_h1, timings['garud_h'] = predict_garud(data, benchmark=True)
    else:
        ihs_maxabs = predict_ihs_max(data)
        tajima_d = predict_td(data)
        garud_h1, garud_h12, garud_h123, garud_h2_h1 = predict_garud(data)
    #n_columns = predict_n_columns(data) all the same for us

    stats = [ihs_maxabs, tajima_d, garud_h1, garud_h12, garud_h123, garud_h2_h1]
    
    if benchmark:
        return stats, timings
    return stats

def prep_our_data(matrix):
    """Convert out data into the right format for functions below"""
    haplos = np.expand_dims(matrix[:,:,0], axis=-1)
    intersnp = matrix[:,:,1][0] * global_vars.L
    positions = np.cumsum(intersnp)
    return [haplos, positions]

def predict_ihs_max(data: List[np.ndarray], benchmark=False) -> float:
    """Computes ihs statistic values from genetic data and then returns the maximum abs value of the statistics

    Parameters
    ----------
    data List[np.ndarray]: [genetic data, genetic positions]
    benchmark (bool): Whether to return timing information

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    if benchmark:
        start_time = time.time()
        
    if not isinstance(data, list):
        raise Exception('The ihs test statistic has multiple inputs')
    genetic_data = data[0][:, :, 0]
    positions = data[1][:]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int32)
    h1 = allel.HaplotypeArray(haplos)
    ihs = allel.ihs(h1, pos=positions, include_edges=True, use_threads=False)
    output = float(np.nanmax(np.abs(ihs)))

    if benchmark:
        end_time = time.time()
        return output, end_time - start_time
    return output

def predict_td(data: List[np.ndarray], benchmark=False) -> float:
    """Computes tajima d test statistic of genetic data

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    benchmark (bool): Whether to return timing information

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    if benchmark:
        start_time = time.time()
        
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int32)
    h1 = allel.HaplotypeArray(haplos)
    ac = h1.count_alleles()
    output = allel.tajima_d(ac)
    
    if benchmark:
        end_time = time.time()
        return output, end_time - start_time
    return output

def predict_nsl(data: List[np.ndarray], benchmark=False) -> float:
    """Computes nsl test statistic of genetic data

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    benchmark (bool): Whether to return timing information

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    if benchmark:
        start_time = time.time()
        
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int32)
    h1 = allel.HaplotypeArray(haplos)
    nsl = allel.nsl(h1)
    output = np.nanmax(np.abs(nsl))
    
    if benchmark:
        end_time = time.time()
        return output, end_time - start_time
    return output

def predict_garud(data: List[np.ndarray], benchmark=False) -> tuple[float, float, float, float]:
    """Computes garud's test statistics

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    benchmark (bool): Whether to return timing information

    Returns
    -------
    h1 (float): Garud's H1
    h12 (float): Garud's H12
    h123 (float): Garud's H123
    h2_h1 (float): H2 / H1
    timing (float): Time taken to compute statistics, optional
    """
    if benchmark:
        start_time = time.time()
        
    genetic_data = data[0][:, :, 0]
    haplos = np.swapaxes(genetic_data, 0, 1).astype(np.int32)
    h = allel.HaplotypeArray(haplos)
    h1, h12, h123, h2_h1 = allel.garud_h(h)
    
    if benchmark:
        end_time = time.time()
        return h1, h12, h123, h2_h1, end_time - start_time
    return h1, h12, h123, h2_h1

def predict_n_columns(data: List[np.ndarray], benchmark=False) -> float:
    """Computes statistic based on number of columns in image

    Parameters
    ----------
    data (np.ndarray): Genetic data to compute test statistic of
    benchmark (bool): Whether to return timing information

    Returns
    -------
    output (float): Returns the statistic of the data

    """
    if benchmark:
        start_time = time.time()
        
    genetic_data = data[0][:, :, 0]
    output = genetic_data.shape[1] # SM: removed negative sign
    
    if benchmark:
        end_time = time.time()
        return output, end_time - start_time
    return output
