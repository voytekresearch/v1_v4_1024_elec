
"""
Compute spectral power of LFP data using the multitaper method.

"""

# imports - general
import numpy as np
import os

# imports - custom
from paths import EXTERNAL_PATH
from info import SESSIONS, FS, EPOCH_TIMES, N_JOBS, N_ARRAYS, N_CHANS
from utils import compute_tfr

# Settings
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
        path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_epochs/{session}'
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
            time = np.linspace(*EPOCH_TIMES, lfp.shape[-1], endpoint=False)
            
            # compute PSD
            tfr, freq = compute_tfr(lfp, FS, FREQS, freq_spacing=FREQ_SPACING, 
                                    n_jobs=N_JOBS, output='avg_power')

            # save results
            fname_out = fname_in.replace('.npy', '.npz')
            np.savez(f"{path_out}/{fname_out}", tfr=tfr, time=time, freq=freq)

        # aggregate results across files (arrays) and save
        print(f"    Aggregating results across arrays...")
        _, n_freqs, n_times = tfr.shape
        spec_shape = [N_CHANS*N_ARRAYS, n_freqs, n_times]
        spectrogram = np.zeros(spec_shape)
        for i_array in range(N_ARRAYS):
            nsp_idx = int(np.ceil((i_array+1)/2))
            data_in = np.load(fr"{path_out}\NSP{nsp_idx}_array{i_array+1}_LFP.npz")
            spectrogram[i_array*N_CHANS : (i_array+1)*N_CHANS] = data_in['tfr']

        # save results
        np.savez(f"{path_out_j}/{session}_lfp.npz", spectrogram=spectrogram, 
                 time=time, freq=freq)


if __name__ == "__main__":
    main()
    