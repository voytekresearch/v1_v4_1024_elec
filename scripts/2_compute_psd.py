
"""
Compute power spectral density (PSD).

"""

# Imports
import numpy as np
import os
# from neurodsp.spectral import compute_spectrum
from mne.time_frequency import psd_array_multitaper

# Settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024/'
SESSIONS = ['A_SNR_140819','L_SNR_250717'] # List of essions to analyze
FS = 500 # sampling frequency (Hz)
# F_RANGE = None # frequency range for spectral analysis. None defaults to 2 cycles at lowest freq and Nyquist freq
N_JOBS = -1 # number of jobs for parallel processing
N_ARRAYS = 16 # number of arrays per session

def main():
    # identify/create directories
    path_out_j = f'{PROJECT_PATH}/data/lfp/lfp_psd/sessions'
    if not os.path.exists(path_out_j): os.makedirs(path_out_j)
        
    # loop through sessions of interest
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_psd/{session}'
        if not os.path.exists(path_out): os.makedirs(path_out)
            
        # loop through files
        files = os.listdir(path_in)
        for i_file, fname_in in enumerate(files):
            # show progress
            print(f'    Processing file {i_file+1}/{len(files)}: \t{fname_in}')

            # load data
            data_in = np.load(f'{path_in}/{fname_in}')
            
            # compute PSD
            # freq, spectra = compute_spectrum_3d(data_in, FS, f_range=F_RANGE) 
            spectra, freq = psd_array_multitaper(data_in, FS, n_jobs=N_JOBS, verbose=False)

            # save results
            fname_out = fname_in.replace('.npy', '.npz')
            np.savez(f"{path_out}/{fname_out}", spectra=spectra, freq=freq)

        # aggregate results across files (arrays) and save
        print(f"    Aggregating results across arrays...")
        spectra_shape = [spectra.shape[0], spectra.shape[1]*N_ARRAYS, spectra.shape[2]]
        for epoch in ['pre', 'post']:
            spectra = np.zeros(spectra_shape)
            for i_array in range(N_ARRAYS):
                nsp_idx = int(np.ceil((i_array+1)/2))
                data_in = np.load(fr"{path_out}\NSP{nsp_idx}_array{i_array+1}_LFP_{epoch}.npz")
                spectra_i = data_in['spectra']
                spectra[:, i_array*spectra_i.shape[1]:(i_array+1)*spectra_i.shape[1]] = spectra_i

            # save results
            np.savez(f"{path_out_j}/{session}_lfp_{epoch}.npz", spectra=spectra, freq=freq)


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
    