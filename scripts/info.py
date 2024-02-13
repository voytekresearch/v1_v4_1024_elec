""" Dataset information """

# Dataset details
SESSIONS = ['A_SNR_041018','A_SNR_140819','L_SNR_250717']

# LFP data details
FS = 500 # LFP sampling frequency (Hz)
EPOCH_TIMES = [-0.3, 0.7] # start and end times of the LFP epoch (s)
IDX_ZERO = 150 # index of zero in the lfp epoch (stimulus onset)
N_ARRAYS = 16 # number of arrays per session
N_CHANS = 64 # number of channels per array

# analysis settings
N_JOBS = -1 # number of jobs for parallel processing
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [2, 12],    # (default: (0.5, 12.0)) should be >= 2*frequency_resolution
    'min_peak_height'   :   0.1,        # (default: 0.0)
    'max_n_peaks'       :   3,          # (default: inf)
    'peak_threshold'    :   2.0,        # (default: 2.0)
    'aperiodic_mode'    :   'knee',     # (default: 'fixed')
    'verbose'           :   False}      # (default: True)

BANDS = {
    'alpha'    :   [8, 16],
    'beta'     :   [16, 40],
    'gamma'    :   [40, 100]
    }
