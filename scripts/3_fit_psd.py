"""
compute Spectral Parametrization for all files in a given folder (session) 

"""

# imports - general
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, fit_models_3d, Bands

# imports - custom
from info import *

# analysis settings
N_ARRAYS = 16
N_CHANS = 64
AVERAGE_TRIALS = True # average across trials before fitting
FREQ_RANGE = [1, 100] 
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [4, 100], # default: (0.5, 12.0)) - reccomends at least frequency resolution * 2
    'min_peak_height'   :   0.1, 
    'max_n_peaks'       :   3, # (default: inf)
    'peak_threshold'    :   2.0, # (default: 2.0)
    'aperiodic_mode'    :   'knee',
    'verbose'           :   False}

bands = Bands(
    {'alpha'    :   [8, 16],
    'beta'     :   [16, 40],
    'gamma'    :   [40, 100]})

def main():
    # identify/create directories
    path_in = f'{PROJECT_PATH}/data/lfp/lfp_psd/sessions'
    path_out = f'{PROJECT_PATH}/data/lfp/lfp_params/'
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    # loop through files
    files = os.listdir(path_in)
    dfs = []
    for i_file, fname_in in enumerate(files):
        # show progress
        print(f'Processing file {i_file+1}/{len(files)}: \t{fname_in}')

        # load data
        data_in = np.load(f'{path_in}/{fname_in}')
    
        # initialize SpecParam
        fg = SpectralGroupModel(**SPEC_PARAM_SETTINGS)

        # # parameterize spectra
        if AVERAGE_TRIALS:
            fg.fit(data_in['freq'], np.mean(data_in['spectra'], axis=0), \
                freq_range=FREQ_RANGE)
        else:
            fit_models_3d(fg, data_in['freq'], data_in['spectra'], \
                freq_range=FREQ_RANGE, n_jobs=N_JOBS)

        # save results
        fname_out = fname_in.replace('.npz', '')
        fg.save(f"{path_out}/{fname_out}", save_results=True, save_settings=True)
        
        # create dataframe with all sessions combined
        df_specparam = fg.to_df(bands)

        # get session IDs
        if i_file%2==0:
            session = fname_in.replace('_lfp_pre.npz', '')
            epoch = 'pre'

        else:
            session = fname_in.replace('_lfp_post.npz', '')
            epoch = 'post' 

        # create a df with session data
        data = {
                'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
                'channel'   :   np.concatenate([np.arange(N_CHANS)] * N_ARRAYS),
                'chan_idx'  :   np.arange(N_ARRAYS*N_CHANS),
                'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS) + 1,
                'epoch'     :   np.repeat(epoch, N_ARRAYS*N_CHANS)}

        df_data = pd.DataFrame(data=data)
        
        df_sess = pd.concat([df_data, df_specparam], axis=1)

        # add dfs to list of dfs for concating
        dfs.append(df_sess)
    
    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{PROJECT_PATH}/data/results/lfp_spectral_params_bands.csv')



if __name__ == "__main__":
    main()

