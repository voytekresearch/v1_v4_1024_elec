"""
compute Spectral Parametrization for all files in a given folder (session) 

"""

# imports - general
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, Bands

# imports - custom
from paths import EXTERNAL_PATH
from info import N_ARRAYS, N_CHANS
from settings import N_JOBS, BANDS, SPECPARAM_SETTINGS

def main():
    # identify/create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions'
    path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_params'
    path_results = f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)
    if not os.path.exists(path_results): os.makedirs(path_results)
        
    # loop through files
    files = os.listdir(path_in)
    dfs = []
    for i_file, fname_in in enumerate(files):
        # show progress
        session = fname_in.replace('_lfp.npz', '')
        print(f'Processing file {i_file+1}/{len(files)}: \t{session}')

        # load data
        data_in = np.load(f'{path_in}/{fname_in}')

        # set indices
        idx_pre = data_in['time']<0
        idx_post = (data_in['time']>0) & (data_in['time']<0.3)

        # loop through pre- and post-stimulus spectrograms
        for idx, epoch in zip([idx_pre, idx_post], ['pre', 'post']):
            
            # average over time
            tfr = np.mean(data_in['spectrogram'][..., idx], axis=-1)

            # parameterize spectra
            fg = SpectralGroupModel(**SPECPARAM_SETTINGS)
            fg.fit(data_in['freq'], tfr, n_jobs=N_JOBS)

            # save specparam results object
            fname_out = fname_in.replace('.npz', f'_{epoch}')
            fg.save(f"{path_out}/{fname_out}", save_results=True, 
                    save_settings=True)
        
            # create dataframe of results
            df_specparam = fg.to_df(Bands(BANDS))
            df_data = pd.DataFrame({
                'session'   :   np.repeat(session, N_ARRAYS*N_CHANS),
                'channel'   :   np.concatenate([np.arange(N_CHANS)] * N_ARRAYS),
                'chan_idx'  :   np.arange(N_ARRAYS*N_CHANS),
                'array'     :   np.repeat(np.arange(N_ARRAYS), N_CHANS) + 1,
                'epoch'     :   np.repeat(epoch, N_ARRAYS*N_CHANS)})
            df_sess = pd.concat([df_data, df_specparam], axis=1)

            # add df to list
            dfs.append(df_sess)
    
    # join results DFs across sessions and save
    dfs = pd.concat(dfs, ignore_index=True)
    dfs.to_csv(fr'{path_results}/lfp_spectral_params.csv')

if __name__ == "__main__":
    main()
