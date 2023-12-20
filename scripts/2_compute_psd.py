
"""
Compute spectral power of LFP data using the multitaper method.

"""

# imports - general
import numpy as np
import os
from mne.time_frequency import tfr_array_multitaper

# imports - custom
from info import *

# Settings
SESSIONS = ['A_SNR_041018','A_SNR_140819','L_SNR_250717'] # List of essions to analyze
FREQS = [4, 100, 128] # Frequency vector (start, stop, n_freqs)
FREQ_SPACING = 'lin' # Frequency spacing ('lin' or 'log')

def main():
    # identify/create directories
    path_out_j = f'{PROJECT_PATH}/data/lfp/lfp_tfr/sessions'
    if not os.path.exists(path_out_j): os.makedirs(path_out_j)
        
    # loop through sessions of interest
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_tfr/{session}'
        if not os.path.exists(path_out): os.makedirs(path_out)
            
        # loop through files
        files = os.listdir(path_in)
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
        for epoch in ['pre', 'post']:
            spectrogram = np.zeros(spec_shape)
            for i_array in range(N_ARRAYS):
                nsp_idx = int(np.ceil((i_array+1)/2))
                data_in = np.load(fr"{path_out}\NSP{nsp_idx}_array{i_array+1}_LFP_{epoch}.npz")
                tfr_i = data_in['tfr']
                spectrogram[:, i_array*tfr_i.shape[1]:(i_array+1)*tfr_i.shape[1]] = tfr_i

            # save results
            np.savez(f"{path_out_j}/{session}_lfp_{epoch}.npz", spectrogram=spectrogram, freq=freq)


def compute_tfr(lfp, fs, freqs, freq_spacing='lin', time_window_length=0.5, freq_bandwidth=4, 
                n_jobs=-1, decim=1, verbose=False):
    """
    Compute time-frequency representation (TFR) of LFP data.

    Parameters
    ----------
    lfp : 3d array
        LFP data (trials x channels x samples).
    fs : int
        Sampling frequency.
    freqs : 1d array
        Frequency vector (start, stop, n_freqs).
    """

    # define hyperparameters
    if freq_spacing == 'lin':
        freq = np.linspace(*freqs)
    elif freq_spacing == 'log':
        freq = np.logspace(*np.log10(freqs[:2]), freqs[2])
    n_cycles = freq * time_window_length # set n_cycles based on fixed time window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_array_multitaper(lfp, sfreq=fs, freqs=freq, n_cycles=n_cycles, 
                                time_bandwidth=time_bandwidth, output='power', n_jobs=n_jobs,
                                decim=decim, verbose=verbose)

    return tfr, freq


if __name__ == "__main__":
    main()
    