'''
Epoch lfp of select files
'''

# general
import numpy as np
import os

# custom
import sys
sys.path.append("../")
from scripts.utils import epoch_nix

# settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024'
SESSIONS = ['A_SNR_140819'] # List of sessions to analyze
FS = 500 # sampling frequency (Hz)


def main():
    for session in SESSIONS:
        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/dataset/{session}/lfp/nix'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # load and epoch data
        files = os.listdir(path_in)
        for file in files :
            print(f"Analyzing: \t{file}")
            lfp = epoch_nix(f"{path_in}/{file}")

            lfp_pre = lfp[:,:,:250]
            lfp_post = lfp[:,:,250:]

        # save data to file
        np.save(f"{path_out}/{file.replace('.nix','.npy')}", lfp)
        np.save(f"{path_out}/{file.replace('.nix','_pre.npy')}", lfp_pre)
        np.save(f"{path_out}/{file.replace('.nix','_post.npy')}", lfp_post)

if __name__ == "__main__":
    main()
