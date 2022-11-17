
"""
compute power spectral density (PSD) of LFP for a given session. combine results 
for all arrays into single .npz file.

"""

# general imports
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# lab imports
from neurodsp.spectral import compute_spectrum

# Dataset details
FS = 500 # sampling frequency (Hz)
N_ARRAYS = 16 # number of arrays
N_CHANNELS = 64 # number of channels per array
# SESSIONS = []

# Settings
PATH = 'G:/Shared drives/v1_v4_1024/'
SESSION = 'A_SNR_041018'

# analysis settings
F_RANGE = [1, FS/2] # frequency range for spectral analysis - skip freq=0

def main():
    """
    load data file for each array and compute the PSD of the LFP signals
    """

    # initialize variables (i.e. create empty arrays to be filled with results for each file)
    data_in = loadmat(PATH + f'data/{SESSION}/lfp/NSP1_array1_LFP.mat')
    freq, _ = compute_spectrum(data_in['lfp'].T, FS, f_range=F_RANGE)
    psd = np.zeros([N_ARRAYS*N_CHANNELS, len(freq)]) # shape: n channels (1024) x n frequencies 

    # loop through files
    idx_nsp = 1
    for i_array in range(1, N_ARRAYS+1):
        # display progress
        print(f"Analyzing file#: \t{i_array}")

        # skip missing file
        if i_array == 15: continue

        # import data
        fname_in = PATH + f'data/{SESSION}/lfp/NSP{idx_nsp}_array{i_array}_LFP.mat'
        data_in = loadmat(fname_in)

        # compute power spectrum
        _, spectrum_i = compute_spectrum(data_in['lfp'].T, FS, f_range=F_RANGE) # here, we use '.T' to transpose the array so the first dimension is channels
        psd[(i_array-1)*N_CHANNELS:i_array*N_CHANNELS] = spectrum_i # save results of each file to a single variable

        # increment
        if i_array % 2 == 0:
            idx_nsp += 1

    # save results
    np.savez(PATH + f'/data/results/{SESSION}_lfp_spectra', psd=psd, freq=freq) 

if __name__ == "__main__":
    main()
    