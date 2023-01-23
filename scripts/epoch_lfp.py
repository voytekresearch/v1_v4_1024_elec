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
PATH_IN = 'G:/Shared drives/v1_v4_1024/data/dataset/A_SNR_140819/lfp/nix'
PATH_OUT = 'G:/Shared drives/v1_v4_1024/data/lfp/lfp_epochs/A_SNR_140819'
FS = 500 # sampling frequency (Hz)


def main():

    # load and epoch data
    files = os.listdir(PATH_IN)
    for file in files :
        print(f"Analyzing: \t{file}")
        lfp = epoch_nix(f"{PATH_IN}/{file}")

        lfp_pre = lfp[:,:,:250]
        lfp_post = lfp[:,:,250:]

        # save data to file
        np.save(f"{PATH_OUT}/{file.replace('.nix','.npy')}", lfp)
        np.save(f"{PATH_OUT}/{file.replace('.nix','_pre.npy')}", lfp_pre)
        np.save(f"{PATH_OUT}/{file.replace('.nix','_post.npy')}", lfp_post)

if __name__ == "__main__":
    main()
