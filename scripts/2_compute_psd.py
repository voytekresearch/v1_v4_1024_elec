
"""
Compute spectral power of LFP data using the multitaper method.

"""

# imports - general
import numpy as np
import os

# imports - custom
from paths import PROJECT_PATH, EXTERNAL_PATH
from info import *
from utils import compute_tfr

# Settings
SESSIONS = ['A_SNR_041018','A_SNR_140819','L_SNR_250717'] # List of essions to analyze
FREQS = [4, 100, 128] # Frequency vector (start, stop, n_freqs)
FREQ_SPACING = 'lin' # Frequency spacing ('lin' or 'log')

def main():
    # identify/create directories
    path_out_j = f'{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions'
    if not os.path.exists(path_out_j): os.makedirs(path_out_j)
        
    # loop through sessions of interest
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_tfr/{session}'
        if not os.path.exists(path_out): os.makedirs(path_out)
            
        # loop through files
        files = os.listdir(path_in)
        files = [f for f in files if not "pre" in f and not "post" in f]
        for i_file, fname_in in enumerate(files):
            # show progress
            print(f'    Processing file {i_file+1}/{len(files)}: \t{fname_in}')

            # load data
            lfp = np.load(f'{path_in}/{fname_in}')
            
            # compute PSD
            tfr, freq = compute_tfr(lfp, FS, FREQS, freq_spacing=FREQ_SPACING, 
                                        n_jobs=N_JOBS)

            # save results
            fname_out = fname_in.replace('.npy', '.npz')
            np.savez(f"{path_out}/{fname_out}", tfr=tfr, freq=freq)

        # aggregate results across files (arrays) and save
        print(f"    Aggregating results across arrays...")
        n_trials, n_channels, n_freqs, n_times = tfr.shape
        spec_shape = [n_trials, n_channels*N_ARRAYS, n_freqs, n_times]
        spectrogram = np.zeros(spec_shape)
        for i_array in range(N_ARRAYS):
            nsp_idx = int(np.ceil((i_array+1)/2))
            data_in = np.load(fr"{path_out}\NSP{nsp_idx}_array{i_array+1}_LFP.npz")
            tfr_i = data_in['tfr']
            spectrogram[:, i_array*tfr_i.shape[1]:(i_array+1)*tfr_i.shape[1]] = tfr_i

        # save results
        np.savez(f"{path_out_j}/{session}_lfp.npz", spectrogram=spectrogram, freq=freq)


if __name__ == "__main__":
    main()
    