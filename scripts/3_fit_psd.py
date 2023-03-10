"""
compute Spectral Parametrization for all files in a given folder (session) 

"""

# Imports
import os
import numpy as np
import pandas as pd
from fooof import FOOOFGroup, fit_fooof_3d

# Settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024/'
FS = 500 # sampling frequency (Hz)
AVERAGE_TRIALS = True # average across trials before fitting

# SpecParam settings
FREQ_RANGE = [1, 100] 
N_JOBS = -1 # number of jobs for parallel processing
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [4, 20], # default: (0.5, 12.0)) - reccomends at least frequency resolution * 2
    'min_peak_height'   :   0.1, 
    'max_n_peaks'       :   4, # (default: inf)
    'peak_threshold'    :   2.0, # (default: 2.0)
    'aperiodic_mode'    :   'knee',
    'verbose'           :   False}

def main():
    # identify/create directories
    path_in = f'{PROJECT_PATH}/data/lfp/lfp_psd/sessions'
    path_out = f'{PROJECT_PATH}/data/lfp/lfp_params/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    # loop through files
    files = os.listdir(path_in)
    for i_file, fname_in in enumerate(files):
        # show progress
        print(f'Processing file {i_file+1}/{len(files)}: \t{fname_in}')

        # load data
        data_in = np.load(f'{path_in}/{fname_in}')
    
        # initialize SpecParam
        fg = FOOOFGroup(**SPEC_PARAM_SETTINGS)

        # # parameterize spectra
        if AVERAGE_TRIALS:
            fg.fit(data_in['freq'], np.mean(data_in['spectra'], axis=0), \
                freq_range=FREQ_RANGE)
        else:
            fit_fooof_3d(fg, data_in['freq'], data_in['spectra'], \
                freq_range=FREQ_RANGE, n_jobs=N_JOBS)

        # save results
        fname_out = fname_in.replace('.npz', '.csv')
        fg.save(f"{path_out}/{fname_out}", save_results=True, save_settings=True)


if __name__ == "__main__":
    main()

