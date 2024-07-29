""" Dataset information """

# Dataset details -- temporarily removed 'A_SNR_041018'
SESSIONS = ['A_SNR_140819','L_SNR_250717']

# LFP data details
FS = 500 # LFP sampling frequency (Hz)
EPOCH_TIMES = [-0.3, 0.7] # start and end times of the LFP epoch (s)
IDX_ZERO = 150 # index of zero in the lfp epoch (stimulus onset)
N_ARRAYS = 16 # number of arrays per session
N_CHANS = 64 # number of channels per array
TOTAL_CHANS = N_ARRAYS * N_CHANS # total number of channels per session
