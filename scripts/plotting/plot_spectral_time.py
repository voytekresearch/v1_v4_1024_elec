"""
This script plots the results of SpectralTimeModel

"""

# imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# imports - lab <development>
from specparam import SpectralTimeEventModel, Bands

# imports - custom
import sys
sys.path.append("code")
from paths import EXTERNAL_PATH
from info import SESSIONS, FS, N_ARRAYS, IDX_ZERO, N_CHANS
from settings import SPECPARAM_SETTINGS, BANDS

# settings
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # identify/create directories
    path_in = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions"
    path_out = f"{EXTERNAL_PATH}/data/lfp/SpectralTimeModel"
    if not os.path.exists(path_out): os.makedirs(path_out)

    # fig_path = f'{EXTERNAL_PATH}/figures/SpectralTimeModel'
    # if not os.path.exists(fig_path): os.makedirs(fig_path)

    path_results = f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_results): os.makedirs(path_results)

    # loop over sessions
    dfs = []
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # load data
        fname = f"{path_in}/{session}_lfp.npz"
        print(fname)
        data_in = np.load(fname)
        print(data_in.files)

        # set variables
        tfr = data_in['spectrogram']
        time = data_in['time']
        freq = data_in['freq']
        print(f"Shape tfr:\t{tfr.shape}")
        print(f"Shape time:\t{time.shape}")
        print(f"Shape freq:\t{freq.shape}")

        # apply SpectralTimeModel
        spec = tfr
        stm = SpectralTimeEventModel(**SPECPARAM_SETTINGS)
        stm.fit(freq, spec)
        # stm.plot(save_fig=True, file_name=f"{session}_spectra.png", file_path=fig_path)

        # save specparam Time Model results object
        stm.save(f'{path_out}/{fname.replace('.npz', '')}', save_results=True, save_settings=True)

        # create a dataframe for reuslts
        df_stm = stm.to_df(Bands(BANDS))
        df_data = pd.DataFrame({
            'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
            'channel'   :   np.concatenate([np.arange(N_CHANS)] * N_ARRAYS),
            'chan_idx'  :   np.arange(N_ARRAYS*N_CHANS),
            'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS) + 1,})
        df_sess = pd.concat([df_data, df_stm], axis=1)

        # add df to list
        dfs.append(df_sess)
    
    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{path_results}/lfp_stm_params.csv')


if __name__ == "__main__":
    main()
