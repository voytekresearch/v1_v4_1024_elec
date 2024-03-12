"""
This script plots the results of SpectralTimeModel

"""

# imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt

# imports - lab <development>
from specparam import SpectralTimeModel

# imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH, EXTERNAL_PATH
from info import SESSIONS, FS, N_ARRAYS, IDX_ZERO
from settings import SPECPARAM_SETTINGS

# settings
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # identify/create directories
    path_in = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions"
    path_out = f"{EXTERNAL_PATH}/figures/SpectralTimeModel"
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # loop over sessions
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
        spec = tfr[0]
        stm = SpectralTimeModel(**SPECPARAM_SETTINGS)
        stm.fit(freq, spec)
        stm.plot(save_fig=True, file_name=f"{session}_spectra.png", file_path=path_out)


if __name__ == "__main__":
    main()
