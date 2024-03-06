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
from utils import crop_tfr
from paths import EXTERNAL_PATH

# settings
DURATION = 0.3 # duration of pre- and post-stimulus epochs (in seconds)

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

        # crop tfr to epochs
        tfr_pre, _ = crop_tfr(tfr, time, [-DURATION, 0])
        tfr_post, _ = crop_tfr(tfr, time, [0, DURATION])

        # compute average power spectrum and save
        psd_pre = np.mean(tfr_pre, axis=-1)
        psd_post = np.mean(tfr_post, axis=-1)

        # save results
        np.savez(f"{path_out}/{file.replace('.npz', '_pre.npz')}", 
                    spectra=psd_pre, freq=freq)
        np.savez(f"{path_out}/{file.replace('.npz', '_post.npz')}", 
                    spectra=psd_post, freq=freq)
            

if __name__ == "__main__":
    main()
    