""""
Aggregate spectral parameters from all sessions into a dataframe.

"""

# imports
import os
import numpy as np
import pandas as pd
from fooof import FOOOFGroup

# Settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024/'
FS = 500 # sampling frequency (Hz)
N_ARRAYS = 16
N_CHANS = 64

def main():
    # identify/create directories
    path_in = f'{PROJECT_PATH}/data/lfp/lfp_params'
    path_out = f'{PROJECT_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)

    # get session IDs
    files = os.listdir(path_in)
    sessions = [fname.replace('_lfp_post.csv', '') for fname in files]
    sessions = [fname.replace('_lfp_pre.csv', '') for fname in sessions]
    sessions = np.unique(sessions)

    # loop through sessions
    dfs = []

    for session in sessions:
        # create dataframe for session data
        data = {
            'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
            'channel'   :   np.arange(N_ARRAYS*N_CHANS),
            'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS)}
        df_sess = pd.DataFrame(data=data)

        # loop through epochs
        for epoch in ['pre', 'post']:
            # import data
            fg = FOOOFGroup()
            fg.load(fr"G:\Shared drives\v1_v4_1024\data\lfp\lfp_params\{session}_lfp_{epoch}.csv")

            # add exponent anf r-squared to dataframe
            df_sess[f'exp_{epoch}'] = fg.get_params('aperiodic_params', 'exponent')
            df_sess[f'r2_{epoch}'] = fg.get_params('r_squared')
            
            # add to list
            dfs.append(df_sess)

    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{path_out}\lfp_spectral_params.csv')


if __name__ == "__main__":
    main()

