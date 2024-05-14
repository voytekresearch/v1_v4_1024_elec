"""
Apply Spectral Parametrization (SpecParam) to all sessions. The output of
scripts/2_compute_spectra.py is used as input. For each session, the 3D 
spectrogram is averaged over time for the pre- and post-stimulus periods and 
then parameterized. Results are aggregated across sessions and the dataframe
is saved as a CSV file.

NOTE: This script schould be ran sequentially, after steps 1-3 in /scripts.

"""

# imports - general
import os
import numpy as np
import pandas as pd
from specparam import SpectralGroupModel, Bands

# imports - custom
import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from info import N_ARRAYS, N_CHANS
from settings import N_JOBS, BANDS, SPECPARAM_SETTINGS

def main():
    # identify/create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_psd'
    path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_params'
    path_results = f'{EXTERNAL_PATH}/data/results'
    if not os.path.exists(path_out): os.makedirs(path_out)
    if not os.path.exists(path_results): os.makedirs(path_results)
        
    # loop through files
    files = os.listdir(path_in)
    dfs = []
    for i_file, fname_in in enumerate(files):
        f_params = fname_in.split('_')
        session = '_'.join(f_params[:3])
        epoch = f_params[4]

        # show progress
        print(f'Processing file {i_file+1}/{len(files)}: \t{session}')

        # load data
        data_in = np.load(f'{path_in}/{fname_in}')
        spectra = data_in['spectra']

        # parameterize spectra
        fg = SpectralGroupModel(**SPECPARAM_SETTINGS)
        fg.fit(data_in['freq'], spectra, n_jobs=N_JOBS)

        # save specparam results object
        fg.save(f"{path_out}/{fname_in.replace('.npz', '')}", save_results=True, 
                save_settings=True, save_data=True)
    
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
