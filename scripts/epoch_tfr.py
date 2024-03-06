"""
Epoch Cropped Time Frequency Representations (tfr)
"""

# imports - general
import numpy as np
import os

# imports - custom 
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
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        
        # loop over files (arrays; 16 per session)
        files = os.listdir(path_in)
        files = [f for f in files if f.endswith('.npz')]
        for file in files:
            data = np.load(f'{path_in}/{file}')
        
            tfr = data['tfr']
            time = data['time']
            freq = data['freq']

            # crop tfr to epochs
            pre_cropped = crop_tfr(tfr, time, [DURATION[0],0])
            post_cropped = crop_tfr(tfr, time, [0, DURATION[1]])

            # save results
            fname_out = file.replace('.npz', '_XXX.npz')
            np.savez(f"{path_out}/{fname_out.replace('XXX', 'pre')}", tfr=pre_cropped[0], time=pre_cropped[1], freq=freq)
            np.savez(f"{path_out}/{fname_out.replace('XXX', 'post')}", tfr=post_cropped[0], time=post_cropped[1], freq=freq)

if __name__ == "__main__":
    main()
    