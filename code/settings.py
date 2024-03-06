""" Analysis settings"""

# Spectral analysis settings
FREQS = [4, 100, 128] # Frequency vector for spectra (start, stop, n_freqs)
FREQ_SPACING = 'lin' # Frequency spacing for spectra ('lin' or 'log')

# SpecParam settings
N_JOBS = -1 # number of jobs for parallel processing
SPECPARAM_SETTINGS = {
    'peak_width_limits' :   [2, 12],    # (default: (0.5, 12.0)) should be >= 2*frequency_resolution
    'min_peak_height'   :   0.1,        # (default: 0.0)
    'max_n_peaks'       :   3,          # (default: inf)
    'peak_threshold'    :   2.0,        # (default: 2.0)
    'aperiodic_mode'    :   'knee',     # (default: 'fixed')
    'verbose'           :   False}      # (default: True)

BANDS = { # oscillation frequency bands of interest
    'alpha'    :   [8, 16],
    'beta'     :   [16, 40],
    'gamma'    :   [40, 100]
    } 
