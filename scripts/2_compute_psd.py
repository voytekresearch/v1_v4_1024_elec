
"""
compute power spectral density (PSD) for all LFP files in a given folder

"""

# Imports
import numpy as np
import os
# from neurodsp.spectral import compute_spectrum
from mne.time_frequency import psd_array_multitaper

# Settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024/'
SESSIONS = ['L_SNR_250717'] # List of essions to analyze
FS = 500 # sampling frequency (Hz)
# F_RANGE = None # frequency range for spectral analysis. None defaults to 2 cycles at lowest freq and Nyquist freq
N_JOBS = -1 # number of jobs for parallel processing

def main():
    # loop through sessions of interest
    for session in SESSIONS:
        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_psd/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)
            
        # loop through files
        files = os.listdir(path_in)
        for i_file, fname_in in enumerate(files):
            # show progress
            print(f'Processing file {i_file+1}/{len(files)}: \t{fname_in}')

            # load data
            data_in = np.load(f'{path_in}/{fname_in}')
            
            # compute PSD
            # freq, spectra = compute_spectrum_3d(data_in, FS, f_range=F_RANGE) 
            spectra, freq = psd_array_multitaper(data_in, FS, n_jobs=N_JOBS)

            # save results
            fname_out = fname_in.replace('.npy', '.npz')
            np.savez(f"{path_out}/{fname_out}", spectra=spectra, freq=freq)


# def compute_spectrum_3d(signal, fs, f_range=None):
#     # set freq range
#     if f_range is None:
#         f_range = [2*len(signal)/fs, fs/2] # 2 cycles at lowest freq and Nyquist freq

#     # loop trough trials
#     spectra = []
#     for i_trial in range(len(signal)):
            
#         # compute spectrum
#         freq, spectra_i = compute_spectrum(signal, fs, f_range=f_range)
#         spectra.append(spectra_i)

#     # convert to array
#     spectra = np.array(spectra)

#     return freq, spectra


if __name__ == "__main__":
    main()
    