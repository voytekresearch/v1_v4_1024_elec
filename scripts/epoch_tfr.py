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
from info import SESSIONS
from paths import EXTERNAL_PATH

def main():
    # set global variable for duration of pre and post stimulus
    DURATION = [-0.3, 0.3]

    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # create directories
        path_in = f'{EXTERNAL_PATH}/data/lfp/lfp_tfr/{session}'
        path_out = f'{EXTERNAL_PATH}//data/lfp/lfp_tfr_epoch/{session}'
        path_psd = f'{EXTERNAL_PATH}//data/lfp/lfp_psd'
        for path in [path_out, path_psd]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # loop over files (arrays; 16 per session)
        files = os.listdir(path_in)
        files = [f for f in files if f.endswith('.npz')]
        for file in files:
            # load data
            data = np.load(f'{path_in}/{file}')
            tfr = data['tfr']
            time = data['time']
            freq = data['freq']

            # crop tfr to epochs
            pre_cropped = crop_tfr(tfr, time, [DURATION[0], 0])
            post_cropped = crop_tfr(tfr, time, [0, DURATION[1]])

            # save results
            fname_out = file.replace('.npz', '_XXX.npz')
            np.savez(f"{path_out}/{fname_out.replace('XXX', 'pre')}", 
                     tfr=pre_cropped[0], time=pre_cropped[1], freq=freq)
            np.savez(f"{path_out}/{fname_out.replace('XXX', 'post')}", 
                     tfr=post_cropped[0], time=post_cropped[1], freq=freq)
            
            # compute average power spectrum and save
            psd_pre = np.mean(pre_cropped[0], axis=-1)
            psd_post = np.mean(post_cropped[0], axis=-1)
            np.savez(f"{path_psd}/{fname_out.replace('XXX', 'pre')}", 
                     psd=psd_pre, freq=freq)
            np.savez(f"{path_psd}/{fname_out.replace('XXX', 'post')}",
                     psd=psd_post, freq=freq)
            

if __name__ == "__main__":
    main()
    