"""
This script fits the SpectralTimeModel to the LFP restults (results of
scripts/pipeline/step2_compute_spectrogram.py)

"""

# imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as timer

# imports - lab <development>
from specparam import SpectralTimeEventModel, Bands

# imports - custom
import sys
sys.path.append("code")
from paths import EXTERNAL_PATH
from info import SESSIONS, N_ARRAYS, N_CHANS
from settings import SPECPARAM_SETTINGS, BANDS, N_JOBS
from time_utils import hour_min_sec

# settings
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # display progress
    t_start = timer()

    # identify/create directories
    path_in = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions"
    path_out = f"{EXTERNAL_PATH}/data/lfp/SpectralTimeModel"
    if not os.path.exists(f"{path_out}"): 
        os.makedirs(f"{path_out}")
    if not os.path.exists(f"{EXTERNAL_PATH}/data/results"): 
        os.makedirs(f"{EXTERNAL_PATH}/data/results")

    # loop over sessions
    dfs = []
    for session in SESSIONS:
        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session: {session}")

        # load data
        fname = f"{session}_lfp.npz"
        data_in = np.load(f"{path_in}/{fname}")

        # set variables
        tfr = data_in['spectrogram']
        freq = data_in['freq']

        # apply SpectralTimeModel
        stm = SpectralTimeEventModel(**SPECPARAM_SETTINGS)
        stm.fit(freq, tfr, n_jobs=N_JOBS)

        # save specparam Time Model results object
        stm.save(f"{path_out}/{fname.replace('.npz', '')}", 
                 save_results=True, save_settings=True)

        # create a dataframe for reuslts
        df_stm = stm.to_df(Bands(BANDS))
        n_bins = tfr.shape[-1]
        df_data = pd.DataFrame({
            'session'   :   np.repeat(session, N_ARRAYS*N_CHANS*n_bins),
            'channel'   :   np.concatenate([np.repeat(np.arange(N_CHANS), 
                                                      tfr.shape[-1])]*N_ARRAYS),
            'chan_idx'  :   np.repeat(np.arange(N_ARRAYS*N_CHANS), n_bins),
            'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS*n_bins) + 1})
        df_sess = pd.concat([df_data, df_stm], axis=1)

        # add df to list
        dfs.append(df_sess)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"\tSession completed in {hour} hour, {min} min, and {sec:0.1f} s")
    
    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(f'{EXTERNAL_PATH}/data/results/lfp_stm_params.csv')

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\nTotal analysis time: {hour} hour, {min} min, and {sec:0.1f} s")


if __name__ == "__main__":
    main()
