'''
Epoch LFP data.

'''

# general
import numpy as np
import os

# custom
import sys
sys.path.append("scripts")
from utils import epoch_nix

# settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024'
SESSIONS = ['A_SNR_140819','L_SNR_250717'] # List of sessions to analyze
FS = 500 # sampling frequency (Hz)
IDX_ZERO = 150 # index of zero in the lfp epoch

def main():
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/dataset/{session}/lfp/nix'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # load and epoch data
        files = os.listdir(path_in)
        for file in files :
            print(f"\t{file}")
            lfp = epoch_nix(f"{path_in}/{file}")

            lfp_pre = lfp[...,:IDX_ZERO]
            lfp_post = lfp[...,IDX_ZERO:]

            # save data to file
            np.save(f"{path_out}/{file.replace('.nix','_pre.npy')}", lfp_pre)
            np.save(f"{path_out}/{file.replace('.nix','_post.npy')}", lfp_post)

if __name__ == "__main__":
    main()
