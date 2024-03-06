"""
This script plots the results of scripts.pipeline.step2_compute_spectrogram.py.

Figure 1: plot the average power spectrum for each array in the dataset. Each 
figure includes the power spectrum for the pre- and post-stimulus epochs.

Figure 2: plot the difference in power spectrum between pre- and post-stimulus
epochs for each electrode in the dataset.

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
from utils import comp_psd_diff
from settings import EPOCH_DURATION
from plots import plot_spectra_2conditions, plot_psd_diff

sys.path.append("scripts")
from pipeline.step3_compute_epoch_psd import compute_epoch_psd

# settings
plt.style.use('mpl_styles/default.mplstyle')

def main():
    # loop through each session
    for session in SESSIONS:
        print(f"Plotting session: {session}")

        # plot average power spectrum for each array
        plot_array_spectra(session)

        # plot spectral change
        plot_spectral_change(session)


def plot_array_spectra(session):
        # identify / create directories
        dir_input = f"{EXTERNAL_PATH}/data/lfp/lfp_tfr/{session}"
        dir_output = f"{EXTERNAL_PATH}/figures/spectra/arrays/{session}"
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        # loop through each array
        files = os.listdir(dir_input)
        for fname in files:
            print(f"\tFile: {fname}")

            # load and unpack
            data_in = np.load(f"{dir_input}/{fname}")
            tfr = data_in['tfr']
            time = data_in['time']
            freq = data_in['freq']

            # split pre- and post-stimulus
            psd_pre, psd_post = compute_epoch_psd(tfr, time, EPOCH_DURATION)

            # plot and save
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_spectra_2conditions(psd_pre, psd_post, freq, ax=ax)
            ax.set_title(f"{session} - {fname.split('_')[1]}")
            fig.savefig(f"{dir_output}/{fname.replace('.npz', '.png')}")
            plt.close()


def plot_spectral_change(session):
    # identify / create directories
    dir_output = f"{EXTERNAL_PATH}/figures/spectra/diff"
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # load data
    data_pre = np.load(f"{EXTERNAL_PATH}/data/lfp/lfp_psd/{session}_lfp_pre.npz")
    data_post = np.load(f"{EXTERNAL_PATH}/data/lfp/lfp_psd/{session}_lfp_post.npz")

    # compute log psd difference
    psd_diff = comp_psd_diff(data_pre['spectra'], data_post['spectra'])

    # plot
    fname_out = f"{dir_output}/psd_diff_{session}.png"
    plot_psd_diff(data_pre['freq'], psd_diff, fname_out=fname_out)


if __name__ == "__main__":
    main()
