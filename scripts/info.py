""" Dataset information """

# LFP data details
FS = 500 # LFP sampling frequency (Hz)
IDX_ZERO = 150 # index of zero in the lfp epoch (stimulus onset)
N_ARRAYS = 16 # number of arrays per session
N_CHANS = 64 # number of channels per array

# analysis settings
N_JOBS = -1 # number of jobs for parallel processing
