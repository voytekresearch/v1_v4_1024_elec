"""
This script plots the results of scripts.pipeline.3_compute_spectra.py

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
from utils import crop_tfr

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


def compute_epoch_psd(tfr, time, duration=0.3):
    """
    Split the TFR into pre- and post-stimulus epochs.
    
    """
    
    psd_pre = np.mean(crop_tfr(tfr, time, [-duration, 0])[0], axis=-1)
    psd_post = np.mean(crop_tfr(tfr, time, [0, duration])[0], axis=-1)

    return psd_pre, psd_post


if __name__ == "__main__":
    main()
