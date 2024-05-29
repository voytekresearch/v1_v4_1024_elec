"""
This script fits the SpectralModel to the LFP PSD results (output of
scripts/pipeline/step2_compute_spectrogram.py)

NOTE: This version modifies _ap_bounds to restrict the aperiodic parameter 
fitting to positive values

"""

# imports - general
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, Bands
from specparam.objs import fit_models_3d

# imports - custom
import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from info import SESSIONS, N_ARRAYS, N_CHANS
from settings import N_JOBS, BANDS

def main():
    # identify/create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_psd'
    path_out = f'{EXTERNAL_PATH}/data/lfp/trial_psd_params'
    path_results = f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)
    if not os.path.exists(path_results): os.makedirs(path_results)
        
    # loop through files
    dfs = []
    for i_session, session in enumerate(SESSIONS):
        # show progress
        print(f'Processing file {i_session+1}/{len(SESSIONS)}: \t{session}')

        # load data
        fname_in = f"{session}_spectra.npz"
        data = np.load(f'{path_in}/{fname_in}')

        # analyze baseline and encoding epochs
        for spectra, epoch in zip([data['spectra_pre'], data['spectra_post']], 
                                  ['pre', 'post']):

            # parametrize spectra with/without knee
            for mode in ['fixed', 'knee']:
 
                # create specparam settings according to mode
                SPECPARAM_SETTINGS = {
                    'peak_width_limits' :   [2, 12],    # (default: (0.5, 12.0)) should be >= 2*frequency_resolution
                    'min_peak_height'   :   0.1,        # (default: 0.0)
                    'max_n_peaks'       :   3,          # (default: inf)
                    'peak_threshold'    :   2.0,        # (default: 2.0)
                    'aperiodic_mode'    :   mode,       # (default: 'fixed')
                    'verbose'           :   False}      # (default: True)
                
                fg = SpectralGroupModel(**SPECPARAM_SETTINGS)
                fg._ap_bounds = ((0,0,0), (np.inf, np.inf, np.inf)) # restrict aperiodic parameters to be positive
                fgs = fit_models_3d(fg, data['freq'], spectra, n_jobs=N_JOBS)

                # save specparam results object
                fgs.save(f"{path_out}/{fname_in.replace('.npz', '')}", save_results=True, 
                        save_settings=True, save_data=True)
        
                # create dataframe of results
                df_specparam = fgs.to_df(Bands(BANDS))
                df_data = pd.DataFrame({
                    'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
                    'channel'   :   np.concatenate([np.arange(N_CHANS)] * N_ARRAYS),
                    'chan_idx'  :   np.arange(N_ARRAYS*N_CHANS),
                    'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS) + 1,
                    'epoch'     :   np.repeat(epoch, N_ARRAYS*N_CHANS),
                    'ap_mode'   :   np.repeat(mode, N_ARRAYS*N_CHANS)})
                
                df_sess = pd.concat([df_data, df_specparam], axis=1)

                # add df to list
                dfs.append(df_sess)
    
    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{path_results}/trial_psd_params_unbounded.csv')

if __name__ == "__main__":
    main()
