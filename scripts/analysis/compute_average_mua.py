"""
Compute average MUA for pre- and post-stimulus epochs, across all sessions.

"""

# imports - standard
import numpy as np
import pandas as pd
from neurodsp.utils import create_times

# imports - custom
import sys
sys.path.append('code')
from paths import EXTERNAL_PATH
from utils import load_nix, epoch_neo_segment
from info import FS, SESSIONS, N_ARRAYS, N_CHANS, EPOCH_TIMES


def main():
    # init
    df_list =[]

    # loop through sessions and arrays
    for session in SESSIONS:
        print(f"\nProcessing session {session}...")

        for i_array in range(N_ARRAYS):
            # get current filename
            nps_idx = (i_array+2) // 2
            fname_in = f"{EXTERNAL_PATH}/V1_v4_1024_electrode_resting_state_data/data/{session}/MUAe/NSP{nps_idx}_array{i_array+1}_MUAe.nix"
            print(f"    {fname_in.split('/')[-1]}")
            
            # initialize data dictionary
            data = {
                'session'   :   np.repeat(session, N_CHANS),
                'array'     :   np.repeat(i_array, N_CHANS) + 1,
                'channel'   :   np.concatenate([np.arange(N_CHANS)])}
            
            # load data
            segment, _ = load_nix(fname_in)

            # epoch data into trials around stimulus times
            epochs = epoch_neo_segment(segment, segment.epochs[0], reset_time=True) 
            time = create_times(epochs.shape[-1]/FS, FS, 
                                       start_val=EPOCH_TIMES[0])

            # average MUA over trials
            mua_mean = np.mean(epochs, axis=0) # channels x time
            
            # find indices for pre- and post-stimulus epochs
            idx_pre = np.logical_and(time>=EPOCH_TIMES[0], time<0)
            idx_post = np.logical_and(time>=0, time<EPOCH_TIMES[1]) 

            # loop through epochs 
            for epoch, indices in zip(['pre', 'post'], [idx_pre, idx_post]):

                # take average over time window
                mua_epoch = np.mean(mua_mean[:, indices], axis=1)

                # store results
                df = pd.DataFrame(data=data)
                df['epoch'] = epoch
                df[f'mua'] = np.mean(mua_epoch)
                df_list.append(df)

    # concatenate results into single dataframe
    df = pd.concat(df_list, ignore_index=True)

    # save results
    df.to_csv(f'{EXTERNAL_PATH}/data/results/mua_df.csv', index=False)


if __name__ == '__main__':
    main()
