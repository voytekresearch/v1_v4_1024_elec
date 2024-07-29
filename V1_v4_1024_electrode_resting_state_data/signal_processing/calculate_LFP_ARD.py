"""Calculate LFP

Calculate the Local Field Potential (LFP) per channel from the raw analog 
signal. The LFP is downsampled to 500 Hz and saved as a .nix and .mat file.
The filtering steps employed by Chen et al. 2022 have been removed.

"""

# imports - general
from utils import anasig_from_array, merge_anasiglist, mark_epochs
import neo
from neo.io.nixio import NixIO
import gc
from scipy.io import savemat
import os
import glob

# imports - custom
import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from info import SESSIONS, N_ARRAYS

# settings 
SESSIONS = ['A_SNR_140819', 'L_SNR_250717']


if __name__ == '__main__':

    # loop through sessions of interest
    for session in SESSIONS:

        # set/create directories
        folder_ns6 = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/raw"
        odmlpath = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/metadata_{session}.odml "
        folder_out = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/LFP_unfiltered"
        if not os.path.exists(folder_out): os.makedirs(folder_out)

        # call all .ns6 files in folder
        ns6_files = glob.glob(os.path.join(folder_ns6, '*.ns6'))

        # loop through arrays
        for i_array in range(1, N_ARRAYS+1):
            print(f"    pre-processing array: {i_array}/{N_ARRAYS}")

            # get paths
            idx_nsp = (i_array-1) // 2  # retrieve ns6 file per array
            path_ns6 = ns6_files[idx_nsp]
            path_lfp = folder_out + f'/NSP{idx_nsp+1}_array{i_array}_LFP.nix' 
        
            # 1. Get the analogsignal of each index (i.e. electrode)
            lfp = []
            for anasig in anasig_from_array(path_ns6, i_array, odmlpath=odmlpath):

                # Downsample signal from 30kHz to 500Hz resolution (factor 60)
                anasig = anasig.downsample(60, ftype='fir')
                gc.collect()

                lfp.append(anasig)

            # Use custom function to merge analogsignals
            lfp = merge_anasiglist(lfp)

            # Create empty neo blocks for new datafiles
            array_block = neo.Block(name='LFP signal from ' + path_ns6)
            array_block.segments = [neo.Segment()]

            # Enrich with analog signal data
            array_block.segments[0].analogsignals = []
            array_block.segments[0].analogsignals.append(lfp)

            # Mark epochs, based on metadata
            array_block = mark_epochs(array_block, odmlpath, eyepath=None)

            # Save to nix
            outFile = NixIO(path_lfp, mode='ow')
            outFile.write(array_block)
            outFile.close()

            # Save to mat
            matdict = {'lfp': lfp.magnitude, **lfp.array_annotations}
            savemat(path_lfp.replace('nix', 'mat'), matdict)
