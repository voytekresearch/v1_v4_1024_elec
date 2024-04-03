"""
This script computes the average power spectrum for the pre- and post-stimulus 
epochs. The spectrogram [or time-frequency representations (tfr)] is loaded,
cropped within the pre- and post-stimulus epochs, and then averaged across time.

"""

# imports - general
import numpy as np
import os

# imports - custom 
import sys
sys.path.append("code")
from paths import EXTERNAL_PATH
from settings import EPOCH_DURATION
from utils import crop_tfr


def main():

    # create directories
    path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_tfr/sessions'
    files = os.listdir(path_in)
    path_out = f'{EXTERNAL_PATH}//data/lfp/lfp_psd'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # loop through each session
    for file in files:

        # display progress
        print(f"\nAnalyzing: {file}")
    
        # load data
        data = np.load(f'{path_in}/{file}')
        tfr = data['spectrogram']
        time = data['time']
        freq = data['freq']

        # compute average power spectrum for pre- and post-stimulus epochs
        psd_pre, psd_post = compute_epoch_psd(tfr, time, EPOCH_DURATION)

        # save results
        np.savez(f"{path_out}/{file.replace('.npz', '_pre.npz')}", 
                    spectra=psd_pre, freq=freq)
        np.savez(f"{path_out}/{file.replace('.npz', '_post.npz')}", 
                    spectra=psd_post, freq=freq)


def compute_epoch_psd(tfr, time, duration):
    """
    Split the TFR into pre- and post-stimulus epochs.
    
    """
    
    psd_pre = np.mean(crop_tfr(tfr, time, [-duration, 0])[0], axis=-1)
    psd_post = np.mean(crop_tfr(tfr, time, [0, duration])[0], axis=-1)

    return psd_pre, psd_post
            

if __name__ == "__main__":
    main()
    