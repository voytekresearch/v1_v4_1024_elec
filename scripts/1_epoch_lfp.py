'''
Epoch LFP data.
NOTE: this script is written for the RHS dataset; the other datasets have
multiple lists of epochs within segement.epochs that must be considered.

'''

# general
import numpy as np
import os

# custom
import sys
sys.path.append("scripts")
from utils import load_nix, epoch_neo_segment

# settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024'
SESSIONS = ['A_SNR_041018','A_SNR_140819','L_SNR_250717'] # List of sessions to analyze
FS = 500 # LFP sampling frequency (Hz)
IDX_ZERO = 150 # index of zero in the lfp epoch (stimulus onset)
DURATION = 0.3 # duration of pre-/post-stimulus epoch (s)

def main():
    # calculate number of samples
    n_samples = int(DURATION * FS)

    # loop over sessions
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{PROJECT_PATH}/data/dataset/{session}/lfp/nix'
        path_out = f'{PROJECT_PATH}/data/lfp/lfp_epochs/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # loop over files (arrays)
        files = os.listdir(path_in)
        for file in files :
            # display progress
            print(f"\t{file}")

            # load and epoch data
            segment, _ = load_nix(f"{path_in}/{file}")
            lfp = epoch_neo_segment(segment, segment.epochs[0], 
                                    reset_time=True)

            # extract pre- and post-stimulus epochs
            lfp_pre = lfp[..., IDX_ZERO-n_samples : IDX_ZERO]
            lfp_post = lfp[..., IDX_ZERO : IDX_ZERO+n_samples]

            # save data to file
            np.save(f"{path_out}/{file.replace('.nix','.npy')}", lfp)
            np.save(f"{path_out}/{file.replace('.nix','_pre.npy')}", lfp_pre)
            np.save(f"{path_out}/{file.replace('.nix','_post.npy')}", lfp_post)

if __name__ == "__main__":
    main()
