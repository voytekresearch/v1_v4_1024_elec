"""
This script plots the results of scripts.pipeline.step2_compute_spectrogram.py.

Figure 1: plot the average power spectrum for each session in the dataset. Each 
figure includes the power spectra for the pre- and post-stimulus epochs.

Figure 2: plot the difference in power spectra between pre- and post-stimulus
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
from plots import plot_spectra_2conditions, plot_psd_diff

# settings
plt.style.use('mpl_styles/default.mplstyle')


def main():

    # identify / create directories
    dir_input = f"{EXTERNAL_PATH}/data/lfp/lfp_psd"
    dir_output = f"{EXTERNAL_PATH}/figures/spectra"
    for folder in ['session_average', 'session_diff']:
        if not os.path.exists(f"{dir_output}/{folder}"):
            os.makedirs(f"{dir_output}/{folder}")

    # loop through each array
    for session in SESSIONS:
        print(f"Plotting sessions: {session}")

        # load spectra
        data = np.load(f"{dir_input}/{session}_spectra.npz")

        # average spectra over trials and compute difference
        spectra_pre = np.nanmean(data['spectra_pre'], axis=0)
        spectra_post = np.nanmean(data['spectra_post'], axis=0)
        psd_diff = spectra_post - spectra_pre

        # plot spectra
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_spectra_2conditions(spectra_pre, spectra_post, 
                                    data['freq'], ax=ax)
        ax.set_title(session)
        fig.savefig(f"{dir_output}/session_average/{session}.png")

        # plot difference in spectra
        fname_out = f"{dir_output}/session_diff/{session}.png"
        plot_psd_diff(data['freq'], psd_diff, fname_out=fname_out)
        plt.close('all')


if __name__ == "__main__":
    main()
