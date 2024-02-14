"""
Compute the power spectral density (PSD) of the LFP for a given session. Combine 
results for all arrays into single .npz file. This script ignores the task
epochs and computes the PSD for the entire session.

"""

# Imports - general
import numpy as np
from scipy.io import loadmat
from neurodsp.spectral import compute_spectrum

# Imports - custom
from paths import EXTERNAL_PATH as PATH
from info import SESSIONS, FS, N_ARRAYS, N_CHANS

# Settings
EPOCH_LENGTH = 1 # sec

# analysis settings
F_RANGE = [1, FS/2] # frequency range for spectral analysis - skip freq=0

def main():
    # compute session psd
    for session in SESSIONS:
        comp_session_psd(session)

        # compute psd for epochs
        comp_epoch_psd(session)

def comp_epoch_psd(session):
    """
    compute psd for epochs of given duration
    """

    # initialize variables (i.e. create empty arrays to be filled with results for each file)
    data_in = loadmat(PATH + f'data/{session}/lfp/NSP1_array1_LFP.mat')
    freq, _ = compute_spectrum(data_in['lfp'].T, FS, f_range=F_RANGE)
    duration = len(data_in['lfp']) / FS
    n_epochs = int(np.floor(duration/EPOCH_LENGTH))
    psd = np.zeros([n_epochs, N_ARRAYS*N_CHANS, len(freq)]) # shape: n epochs x n channels (1024) x n frequencies 

    # loop through files
    idx_nsp = 1
    for i_array in range(1, N_ARRAYS+1):


        # import data
        fname_in = f'{PATH}data/{session}/lfp/NSP{idx_nsp}_array{i_array}_LFP.mat'
        data_in = loadmat(fname_in)

        # loop through epochs
        for i_epoch in range(n_epochs):
            n_samples = EPOCH_LENGTH*FS
            lfp_epoch = data_in['lfp'][i_epoch*n_samples:(i_epoch+1)*n_samples]

            # compute power spectrum
            _, spectrum_i = compute_spectrum(lfp_epoch.T, FS, f_range=F_RANGE) 
            psd[i_epoch, (i_array-1)*N_CHANS:i_array*N_CHANS] = spectrum_i # save results of each file to a single variable

        # increment
        if i_array % 2 == 0:
            idx_nsp += 1

    # save results
    np.savez(f'{PATH}/data/results/{session}_lfp_spectra_{EPOCH_LENGTH}s_epochs', 
             psd=psd, freq=freq) 

def comp_session_psd(session):
    """
    load data file for each array and compute the PSD of the LFP signals
    """

    # initialize variables (i.e. create empty arrays to be filled with results for each file)
    data_in = loadmat(PATH + f'data/{session}/lfp/NSP1_array1_LFP.mat')
    freq, _ = compute_spectrum(data_in['lfp'].T, FS, f_range=F_RANGE)
    psd = np.zeros([N_ARRAYS*N_CHANS, len(freq)]) # shape: n channels (1024) x n frequencies 
    psd_array = np.zeros([N_ARRAYS, len(freq)]) #psd per array

    # loop through files
    idx_nsp = 1
    for i_array in range(1, N_ARRAYS+1):
        # display progress
        print('Computing PSD for session')
        print(f"Analyzing file#: \t{i_array}")

        # import data
        fname_in = PATH + f'data/{session}/lfp/NSP{idx_nsp}_array{i_array}_LFP.mat'
        data_in = loadmat(fname_in)

        # compute power spectrum
        _, spectrum_i = compute_spectrum(data_in['lfp'].T, FS, f_range=F_RANGE) # here, we use '.T' to transpose the array so the first dimension is channels
        psd[(i_array-1)*N_CHANS:i_array*N_CHANS] = spectrum_i # save results of each file to a single variable
        psd_array[(i_array-1)] = np.mean(spectrum_i, axis = 0)

        # increment
        if i_array % 2 == 0:
            idx_nsp += 1

    # save results
    np.savez(PATH + f'/data/results/{session}_lfp_spectra', psd=psd, freq=freq, 
             psd_array=psd_array) 


if __name__ == "__main__":
    main()
    