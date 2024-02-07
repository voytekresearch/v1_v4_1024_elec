'''
Epoch LFP data.

NOTE: this script is written for the RHS dataset; the other datasets have
multiple lists of epochs within segement.epochs that must be considered.

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
        path_in = f'{EXTERNAL_PATH}/data/dataset/{session}/lfp/nix'
        path_out = f'{EXTERNAL_PATH}/data/lfp/lfp_epochs/{session}'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # loop over files (arrays; 16 per session)
        files = os.listdir(path_in)
        for file in files :
            # display progress
            print(f"\t{file}")

            # load data, epoch, and save to file
            segment, _ = load_nix(f"{path_in}/{file}")
            lfp = epoch_neo_segment(segment, segment.epochs[0], 
                                    reset_time=True)
            np.save(f"{path_out}/{file.replace('.nix','.npy')}", lfp)

if __name__ == "__main__":
    main()
