"""Calculate LFP

Calculate the Local Field Potential (LFP) per channel from the raw
analog signal.

Usage:
    python calculate_LFP_ARD.py 

Options:
    -h --help     Show this screen and terminate script.
    --ns6=FOLDER  Path to folder containing all .ns6 data files.
    --odml=FILE   Path to .odml metadata file.
    --array=INT   Number of the analized array.
    --eyesig=FILE Path to the eye signals
    --out=FOLDER    Output folder path.
"""
from docopt import docopt
from utils import anasig_from_array, merge_anasiglist, mark_epochs
import quantities as pq
import neo
from neo.io.nixio import NixIO
import gc
import warnings
import scipy

import os
import glob

import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from info import SESSIONS, N_ARRAYS

if __name__ == '__main__':

    SESSIONS = ['A_SNR_140819','L_SNR_250717']

    # loop through sessions of interest
    for session in SESSIONS:

        # Get arguments
        folder_ns6 = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/raw"
        odmlpath = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/metadata_{session}.odml "
        folder_out = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/LFP_unfiltered"
        if not os.path.exists(folder_out): os.makedirs(folder_out)

        # call all .ns6 files in folder
        ns6_files = glob.glob(os.path.join(folder_ns6, '*.ns6'))

        # loop through arrays
        for i_array in range(1, N_ARRAYS+1):
            print(f"    plotting array: {i_array}/{N_ARRAYS}")

            idx_nsp = (i_array-1) // 2                                         # retrieve ns6 file per array
            path_ns6 = ns6_files[idx_nsp]

            path_lfp = folder_out + f'/NSP{idx_nsp+1}_array{i_array}_LFP.nix'      # define path outh

        
            lfp = []
            # 1. Get the analogsignal of each index (i.e. electrode)
            for anasig in anasig_from_array(path_ns6, i_array, odmlpath=odmlpath):

                ## 2. Filter the signal between 1Hz and 150Hz
                #anasig = butter(anasig, lowpass_freq=150.0*pq.Hz)
                #gc.collect()

                # 3. Downsample signal from 30kHz to 500Hz resolution (factor 60)
                anasig = anasig.downsample(60, ftype='fir')
                gc.collect()

                ## 4. Bandstop filter the 50, 100 and 150 Hz frequencies
                ## Compensates for artifacts from the European electric grid
                #for fq in [50, 100, 150]:
                #    anasig = butter(anasig,
                #                    highpass_freq=(fq + 2)*pq.Hz,
                #                    lowpass_freq=(fq - 2)*pq.Hz)
                #    gc.collect()

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
            scipy.io.savemat(path_lfp.replace('nix', 'mat'), matdict)
