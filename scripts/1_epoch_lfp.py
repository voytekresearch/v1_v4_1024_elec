'''
Epoch LFP data. This script loads the LFP data from the SNR datasets and epochs
the data based on the segment.epochs attribute. The resulting epochs are 3D
arrays with shape (trials x channels x time). The epochs are then saved as .npy
files.

NOTE: this script is written for the SNR datasets; the other datasets (RF/RS) 
have multiple lists of epochs within segement.epochs that must be considered.

'''

# imports- general
import numpy as np
import os

# imports - custom
from utils import load_nix, epoch_neo_segment
from info import SESSIONS
from paths import EXTERNAL_PATH

def main():
    # loop over sessions
    for session in SESSIONS:
        # display progress
        print(f"\nAnalyzing session: {session}")

        # identify/create directories
        path_in = f'{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/lfp'
        path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_epochs/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # loop over files (arrays; 16 per session)
        files = os.listdir(path_in)
        files = [f for f in files if f.endswith('.nix')]
        for file in files:
            # display progress
            print(f"\t{file}")

            # load data, epoch, and save to file
            segment, _ = load_nix(f"{path_in}/{file}")
            lfp = epoch_neo_segment(segment, segment.epochs[0], 
                                    reset_time=True)
            np.save(f"{path_out}/{file.replace('.nix','.npy')}", lfp)

if __name__ == "__main__":
    main()
