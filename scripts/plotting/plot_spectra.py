"""
This script plots the results of scripts.pipeline.step2_compute_spectrogram.py.
So far, this script is written to plot the average power spectrum for each array
in the dataset. Each figure includes the power spectrum for the pre- and
post-stimulus epochs.

"""

# imports - standard
import os
import numpy as np
import matplotlib.pyplot as plt

# imports - custom
import sys
sys.path.append("code")
from paths import EXTERNAL_PATH
from info import SESSIONS
from plots import plot_spectra_2conditions
sys.path.append("scripts")
from step3_compute_epoch_psd import compute_epoch_psd

# settings
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # loop through each session
    for session in SESSIONS:
        print(f"Plotting session: {session}")

        # identify / create directories
        dir_input = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/{session}"
        files = os.listdir(dir_input)
        dir_output = f"{EXTERNAL_PATH}/figures/spectra/{session}"
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        # loop through each array
        for fname in files:
            print(f"\tFile: {fname}")

            # load and unpack
            data_in = np.load(f"{dir_input}/{fname}")
            tfr = data_in['tfr']
            time = data_in['time']
            freq = data_in['freq']

            # split pre- and post-stimulus
            psd_pre, psd_post = compute_epoch_psd(tfr, time)

            # plot and save
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_spectra_2conditions(psd_pre, psd_post, freq, ax=ax)
            ax.set_title(f"{session} - {fname.split('_')[1]}")
            fig.savefig(f"{dir_output}/{fname.replace('.npz', '.png')}")
            plt.close()


if __name__ == "__main__":
    main()
