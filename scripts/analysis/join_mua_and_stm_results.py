"""
Compute average MUA across trials and aggregate with LFP params.

"""

# imports - general
import numpy as np
import pandas as pd
from scipy.signal import decimate

# imports - custom
import sys
sys.path.append("code")
from utils import load_nix, epoch_neo_segment
from paths import EXTERNAL_PATH
from info import SESSIONS, N_ARRAYS, N_CHANS, EPOCH_TIMES


def main():
    # load STM LFP data
    lfp_data = pd.read_csv(fr"{EXTERNAL_PATH}\data\results\lfp_stm_params.csv", 
                           index_col=0)

    # initialize 
    mua = np.array([])

    # loop through sessions
    for session in SESSIONS:
        print(f"session : {session}")

        # loop through arrays
        for i_array in range(1, N_ARRAYS+1):
            idx_nsp = (i_array+1) // 2

            # load data
            fname_in = fr"{EXTERNAL_PATH}\V1_v4_1024_electrode_resting_state_data\data\{session}\MUAe\NSP{idx_nsp}_array{i_array}_MUAe.nix"
            segment, analog_signal = load_nix(fname_in)

            # epoch data into trials around stimulus times
            epochs = epoch_neo_segment(segment, segment.epochs[0], 
                                       reset_time=True) 

            for i_chan in range(N_CHANS):
                # compute average MUA across trials
                chan_mua = epochs[:,i_chan,:].mean(axis=0)

                # downsample MUAe signal from 1000Hz to LFP sampling rate (500Hz)
                mua_times = decimate(chan_mua, 2) 
                
                # append to array
                mua = np.append(mua, mua_times)
            
    data = lfp_data.assign(mua=mua)
    data.to_csv(f'{EXTERNAL_PATH}/data/results/lfp_stm_params_mua.csv', 
                index=False)


if __name__ == "__main__":
    main()
