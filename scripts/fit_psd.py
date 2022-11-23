"""
compute Spectral Parametrization for a given session. 

"""

# imports
import numpy as np
from fooof import FOOOFGroup

# Dataset details
FS = 500 # sampling frequency (Hz)
N_ARRAYS = 16 # number of arrays
N_CHANNELS = 64 # number of channels per array
SESSIONS = 'A_SNR_140819'

# settings
PATH = 'G:/Shared drives/v1_v4_1024/'

# analysis settings
F_RANGE = [1, FS/2] # frequency range for spectral analysis - skip freq=0

def main():
    # load example file
    fname_in = PATH + f'data/results/{SESSIONS}_lfp_spectra.npz'
    data_in = np.load(fname_in)
    
    # removing nan channels
    psd = psd = data_in['psd'][~(data_in['psd'] == 0).all(axis=1)]

    #setting up our model
    specparam_sets = {'peak_width_limits': [2, 8], 'min_peak_height': 0.1}
    freq_range = [1, 40]
    fg = FOOOFGroup(**specparam_sets)

    fg.fit(data_in['freq'], psd, freq_range=freq_range)

    # save results
    fname_out = fname_in.replace('spectra.npz', 'params')
    fg.save(fname_out, save_results = True, save_settings = True)

if __name__ == "__main__":
    main()





