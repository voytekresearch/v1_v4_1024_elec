""""
Aggregate spectral parameters from all sessions into a dataframe.

"""

# imports - general
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel

# imports - custom
from info import N_ARRAYS, N_CHANS
from paths import EXTERNAL_PATH

def main():
    # identify/create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_params'
    path_out = f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)

    # get session IDs
    files = os.listdir(path_in)
    sessions = [fname.replace('_lfp_post.json', '') for fname in files]
    sessions = [fname.replace('_lfp_pre.json', '') for fname in sessions]
    sessions = np.unique(sessions)

    # loop through sessions
    dfs = []

    for session in sessions:
        # create dataframe for session data
        data = {
            'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
            'channel'   :   np.concatenate([np.arange(N_CHANS)] * N_ARRAYS),
            'chan_idx'  :   np.arange(N_ARRAYS*N_CHANS),
            'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS) + 1}

        # loop through epochs
        for epoch in ['pre', 'post']:
            df_sess = pd.DataFrame(data=data)
            df_sess['epoch'] = epoch
            
            # import data
            sm = SpectralGroupModel()
            sm.load(fr"G:\Shared drives\v1_v4_1024\data\lfp\lfp_params\{session}_lfp_{epoch}.json")

            # add exponent and r-squared to dataframe
            for ap_param in ['offset', 'knee', 'exponent']:
                df_sess[ap_param] = sm.get_params('aperiodic_params', ap_param)
            df_sess[f'r2'] = sm.get_params('r_squared')
            
            # add to list
            dfs.append(df_sess)

    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{path_out}\lfp_spectral_params.csv')


if __name__ == "__main__":
    main()

