"""
Apply SpecParam for a given session. Spectrogram is averaged over time for the
entire trial and then parameterized.

"""

# SET: session to analyze
SESSIONS = 'A_SNR_140819'

# imports
import numpy as np
from specparam import SpectralGroupModel

# imports - custom
from paths import EXTERNAL_PATH
from settings import SPECPARAM_SETTINGS, N_JOBS

def main():
    # load example file
    fname_in = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions/{SESSIONS}_lfp.npz"
    data_in = np.load(fname_in)
    
    # average over time
    tfr = np.mean(data_in['tfr'], axis=-1)

    # parameterize spectra
    sm = SpectralGroupModel(**SPECPARAM_SETTINGS)
    sm.fit(data_in['freq'], tfr, n_jobs=N_JOBS)

    # save results
    fname_out = fname_in.replace('spectra.npz', 'params')
    sm.save(fname_out, save_results=True, save_settings=True)

if __name__ == "__main__":
    main()





