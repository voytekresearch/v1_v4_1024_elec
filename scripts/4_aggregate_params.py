""""
Aggregate spectral parameters from all sessions into their individual .csv file 
and in an additional dataframe containing  all sessions
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fooof import FOOOFGroup

# Settings
PROJECT_PATH = 'G:/Shared drives/v1_v4_1024/'
SESSIONS = ['L_SNR_250717', "A_SNR_140819"] # List of sessions to analyze
FS = 500 # sampling frequency (Hz)

def main():
    
    # settings
    N_ARRAYS = 16
    allsessions_list = []
    

    for session in SESSIONS :
        
        idx_nsp = 1
        df_arraylist = []

        for i_array in range(1, N_ARRAYS+1):
            # create dataframe
            df_array = pd.DataFrame(columns = ['session', 'array', 'channel', 'pre_exp', 'post_exp', 'pre_r2', 'post_r2'])

            # import data
            fg_pre = FOOOFGroup()
            fg_post = FOOOFGroup()
            fg_pre.load(fr"G:\Shared drives\v1_v4_1024\data\lfp\lfp_params\{session}\NSP{idx_nsp}_array{i_array}_LFP_pre.csv")
            fg_post.load(fr"G:\Shared drives\v1_v4_1024\data\lfp\lfp_params\{session}\NSP{idx_nsp}_array{i_array}_LFP_post.csv")

            # get exponent and add it to dataframe
            pre_exp = fg_pre.get_params('aperiodic_params', 'exponent')
            post_exp = fg_post.get_params('aperiodic_params', 'exponent')
            df_array['pre_exp'] = pre_exp
            df_array['post_exp'] = post_exp

            #  get r squared and add it to dataframe
            pre_r2 = fg_pre.get_params('r_squared')
            post_r2 = fg_post.get_params('r_squared')
            df_array['pre_r2'] = pre_r2
            df_array['post_r2'] = post_r2
            
            # add session, channel and array
            sess = np.repeat(session, len(pre_exp))
            channel = np.linspace(0, len(pre_exp)-1, len(pre_exp))
            df_array['session'] = sess
            df_array['channel'] = channel
            df_array['array'] = i_array

            # append df_array to params
            df_arraylist.append(df_array)
            # increment
            if i_array % 2 == 0:
                idx_nsp += 1

        # save dataframe per session
        df_session = pd.concat(df_arraylist)
        df_session.to_csv(fr'G:\Shared drives\v1_v4_1024\data\results\{session}_params_df.csv')

        # save all sessions dataframe
        allsessions_list.append(df_session)

    df_allsessions = pd.concat(allsessions_list)
    df_allsessions.to_csv(fr'G:\Shared drives\v1_v4_1024\data\results\allsessions_params_df.csv')


if __name__ == "__main__":
    main()

