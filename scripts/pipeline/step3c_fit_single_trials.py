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
from info import N_ARRAYS, N_CHANS, TOTAL_CHANS
from settings import N_JOBS, BANDS
from time_utils import get_start_time, print_time_elapsed

# settings
SESSIONS = ['A_SNR_041018', 'L_SNR_250717']
BOUND_AP_PARAMS = True # restrict aperiodic parameters to be positive

def main():
    # time it
    t_start = get_start_time()

    # identify/create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_psd'
    path_out =  f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)
        
    # loop through files
    df_list = []
    for i_session, session in enumerate(SESSIONS):
        # show progress
        t_start_s = get_start_time()
        print(f'Processing file {i_session+1}/{len(SESSIONS)}: \t{session}')

        # load data
        fname_in = f"{session}_spectra.npz"
        data = np.load(f'{path_in}/{fname_in}')

        # analyze baseline and encoding epochs
        for spectra, epoch in zip([data['spectra_pre'], data['spectra_post']], 
                                  ['pre', 'post']):

            # parametrize spectra with/without knee
            for mode in ['fixed', 'knee']:
                # show progress
                time_start_i = get_start_time()
                print(f'\tAnalyzing Epoch: {epoch}, Mode: {mode}')
 
                # create specparam settings according to mode
                SPECPARAM_SETTINGS = {
                    'peak_width_limits' :   [2, 12],    # (default: (0.5, 12.0)) should be >= 2*frequency_resolution
                    'min_peak_height'   :   0.1,        # (default: 0.0)
                    'max_n_peaks'       :   3,          # (default: inf)
                    'peak_threshold'    :   2.0,        # (default: 2.0)
                    'aperiodic_mode'    :   mode,       # (default: 'fixed')
                    'verbose'           :   False}      # (default: True)
                
                fg = SpectralGroupModel(**SPECPARAM_SETTINGS)
                if BOUND_AP_PARAMS:
                    fg._ap_bounds = ((0,0,0), (np.inf, np.inf, np.inf)) # restrict aperiodic parameters to be positive
                fgs = fit_models_3d(fg, data['freq'], spectra, n_jobs=N_JOBS)
        
                # create dataframe of results
                df_specparam = pd.concat([fg.to_df(Bands(BANDS)) for fg in fgs], ignore_index=True)
                n_trials = len(fgs)
                df_data = pd.DataFrame({
                    'session'   :   np.repeat(session, N_ARRAYS * N_CHANS * n_trials),
                    'trial'     :   np.concatenate([np.repeat(np.arange(n_trials), TOTAL_CHANS)]),
                    'channel'   :   np.concatenate([np.arange(N_CHANS)] * (N_ARRAYS * n_trials)),
                    'chan_idx'  :   np.concatenate([np.arange(TOTAL_CHANS)] * n_trials),
                    'array'     :   np.concatenate([np.repeat(np.arange(N_ARRAYS), N_CHANS)] * n_trials) + 1,
                    'epoch'     :   np.repeat(epoch, N_ARRAYS * N_CHANS * n_trials),
                    'ap_mode'   :   np.repeat(mode, N_ARRAYS * N_CHANS * n_trials)})
                
                df_sess = pd.concat([df_data, df_specparam], axis=1)

                # add df to list
                df_list.append(df_sess)

                # join results DFs across sessions and save
                dfs = pd.concat(df_list, ignore_index=True)
                dfs.to_csv(fr'{path_out}/trial_psd_params_unbounded.csv')

                # print time elapsed
                print_time_elapsed(time_start_i, "\t\tcondition complete in:")

        # print time elapsed
        print_time_elapsed(t_start_s, "\tfile complete in:")

    # print total time elapsed
    print_time_elapsed(t_start, "Total time elapsed:")

if __name__ == "__main__":
    main()
