
"""
This script computes the spectral power of LFP data using the multitaper method.
The results of scripts/1_epoch_lfp.py are used as input. The results for each
electrode array are saved as .npz files, with the spectral power saved as a 
3D array (electrode x freqs x time). The results are then aggregated across
electrode arrays for each session and saved as a .npz file.

NOTE: This script schould be ran sequentially, after scripts/1_epoch_lfp.py.

"""

# imports - general
import numpy as np
import os
from mne.time_frequency import psd_array_multitaper

# imports - custom
import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from info import SESSIONS, FS
from settings import N_JOBS

# settings
BANDWIDTH = 5.0 # frequency bandwidth of multitaper decomposition


def main():
    # identify/create directories
    path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_psd'
    if not os.path.exists(path_out): os.makedirs(path_out)

    # loop through sessions of interest
    for session in SESSIONS:
        # identify session files
        path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_epochs/{session}'
        files = os.listdir(path_in)

        # loop through epochs
        spectra = dict()
        for epoch in ['pre', 'post']:
            # display progress
            print(f"\nAnalyzing session: {session} ({epoch}-stimulus)")
            
            # initialize: will store results for each electrode array in session
            spec_list = []

            # loop through files (electrode arrays) for each epoch
            files_e = [f for f in files if epoch in f]
            for i_file, fname_in in enumerate(files_e):
                # display progress
                print(f'\tProcessing file {i_file+1}/{len(files_e)}: \t{fname_in}')

                # load data
                lfp = np.load(f'{path_in}/{fname_in}')

                # compute PSD
                spectra_i, freq = psd_array_multitaper(lfp, FS, n_jobs=N_JOBS,
                                                       bandwidth=BANDWIDTH,
                                                       verbose=False)
                spec_list.append(spectra_i)
            
            # aggregate results across files (arrays)
            spectra[epoch] = np.concatenate(spec_list, axis=1)

        # save results for session
        fname_out = f"{path_out}/{session}_spectra.npz"
        np.savez(fname_out, spectra_pre=spectra['pre'], 
                 spectra_post=spectra['post'], freq=freq)


if __name__ == "__main__":
    main()
    