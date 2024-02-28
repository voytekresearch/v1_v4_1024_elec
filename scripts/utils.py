"""
Utility functions

"""

# general imports
import numpy as np

def load_nix(fname):
    """
    Loads a NIX file and returns the first segment and the first analog signal 
    from that segment (Note: this dataset contains only 1 segment and signal per file).

    Parameters
    ----------
    fname : str
        Path to the NIX file.

    Returns
    -------
    segment : Neo Segment
        First Segment contained in the file
    signal : Neo AnalogSignal
        First AnalogSignal cointained in the first Segment.
    """

    # imports
    import neo

    # load nix file
    with neo.NixIO(fname, mode='ro') as io:
        block = io.read_block()
        segment = block.segments[0] # this dataset has 1 segment
        signal = segment.analogsignals[0] # this dataset has 1 signal

    return segment, signal   


def load_odml_event_times(fname):
    """
    Loads event times from an odml file. Event tiems represent the time of 
    stimulus onset for successful trials.

    Parameters
    ----------
    fname : str
        Path to odml file.

    Returns
    -------
    event_times : numpy.ndarray
        An array of event times.
    """

    # imports
    import odml

    # load odml file
    metadata = odml.load(fname)

    # get event times (stimulus onset) for successful trials
    trials = metadata['Recording']['Trials']
    event_times = trials['Successful_trials'].properties['t_stim_on'].values
    event_times = np.array(event_times)

    return event_times


def epoch_neo_segment(segment, epochs, reset_time=True):
    """Split a Neo segment into epochs and reshape to 3D numpy array
    with conventional axis order (trials x channels x time).

    Parameters
    ----------
    segment : neo.Segment
        Neo segment object to be epoched. Must contain epochs.
    epochs : neo.Epoch
        Neo epoch object to be used for epoching.
    reset_time : bool, optional
        Whether to reset the time of the epochs to 0. The default is True.
    Returns
    -------
    signal_epoch : np.ndarray
        Matrix of epochs with shape (trials x channels x time).
    """

    # imports
    from neo.utils import cut_segment_by_epoch

    # use Neo utils to epoch
    seg_lst = cut_segment_by_epoch(segment, epochs, reset_time=reset_time)

    # Reformat to array
    signal_epoch = []
    for seg_i in seg_lst:
        signal_epoch.append(seg_i.analogsignals[0].magnitude)

    # Reshape (trials x channels x time)
    signal_epoch = np.dstack(signal_epoch)
    signal_epoch = np.transpose(signal_epoch, axes=[2,1,0])

    return signal_epoch


def epoch_nix(fname):
    """
    This function loads a NIX file, extracts the first segment and the first
    analog signal from that segment, and then epochs the signal based on the
    event times.

    Parameters
    ----------
    fname : str
        Path to the NIX file.

    Returns
    -------
    lfp : 3D array
        Epochs of the LFP signal with shape (trials x channels x time).
    """
    
    # load nix file
    segment, _ = load_nix(fname)

    # segment epochs
    lfp = epoch_neo_segment(segment, segment.epochs[0], reset_time=True)

    return lfp


def comp_psd_diff(psd_pre, psd_post):
    """
    Compute the difference of log psd.
    
    Parameters
    ----------
    psd_pre : array
        Array of pre-stimulus psd.
    psd_post : array
        Array of post-stimulus psd.

    Returns
    -------
    log_psd_diff : array
        Array of log psd difference.
    
    """
    
    # compute log psd
    log_psd_pre = np.log10(psd_pre)
    log_psd_post = np.log10(psd_post)
    
    # compute difference
    log_psd_diff = log_psd_post - log_psd_pre

    return log_psd_diff


def subtract_baseline(signal, time, t_baseline):
    """Subtracts the mean of the signal in a given time interval from the signal.

    Parameters
    ----------
    signal : array
        Signal array to subtract mean from
    time : array
        Array with the corresponding time points of the signal array
    t_baseline : array
        Array with the starting and ending time points for the time interval

    Returns
    -------
    signal : array
        Signal array with mean of the given time interval subtracted
    """

    baseline = np.logical_and(time>=t_baseline[0], time<=t_baseline[1])
    signal = signal - np.mean(signal[baseline])

    return signal


def compute_erp(signal, time, t_baseline):
    """Compute the event-related potential (ERP) of a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal with shape (n_trials, n_channels, n_samples).
    time : ndarray
        Timepoints associated with the signal.
    t_baseline : list
        Interval of timepoints used to calculate the baseline of each trial.

    Returns
    -------
    erp : ndarray
        Event-related potential with shape (n_channels, n_samples).
    """

    # subtract baseline
    signal_norm = np.zeros_like(signal)
    for i_trial in range(signal.shape[0]):
        for i_channel in range(signal.shape[1]):
            signal_norm[i_trial][i_channel] = subtract_baseline(signal[i_trial][i_channel], time, t_baseline)

    # compute ERP (channel-wise)
    erp = np.mean(signal_norm, axis=0)

    return erp


def comp_erp_params(erp, height_thresh=1, min_distance=10):
    """
    Compute ERP parameters (latency, amplitude, width) using scipy.find_peaks.

    Parameters
    ----------
    erp : array
        event related potential (1D array).
    height_thresh : float, optional
        Height threshold for peak detection. The unit here is standard
        deviations above the mean. The default is 1.
    min_distance : int, optional
        Minimum distance between peaks (in samples). The default is 10.

    Returns
    -------
    params : pandas.DataFrame
        DataFrame containing the parameters of the ERP.

    """
    # imports
    import pandas as pd
    from scipy.signal import find_peaks, peak_widths

    # define height thresold
    height = np.mean(erp) + (height_thresh * np.std(erp))

    # init
    params = pd.DataFrame()

    # use scipy.find_peaks to find parameters
    peaks_pos, _ = find_peaks(erp, height=height, distance=min_distance) 
    peaks_neg, _ = find_peaks(-erp, height=height, distance=min_distance)
    peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))
    order = np.argsort(np.concatenate([peaks_pos, peaks_neg])) #use to sort widths
    params['latency'] = peaks

    # amplitude
    params['amp'] = erp[peaks]

    # find peaks width
    widths_pos, _, _, _ = peak_widths(erp, peaks_pos)
    widths_neg, _, _, _ = peak_widths(-erp, peaks_neg)
    widths = np.concatenate([widths_pos, widths_neg])
    params['widths'] = widths[order]

    return params


def trim_signal(signal, time, epoch, dim=None):
    """
    trim signal in time.

    Parameters
    ----------
    signal : float, array-like
        signal to trim.
    time : float, array-like
        associated time vector to trim.
    epoch : float, array-like or list, length: 2
        time window of interest (start_time, end_time)

    Returns
    -------
    signal : float, array-like
        trimmed signal.
    time : float, array-like
        trimmed time vector.

    """

    mask = (time>=epoch[0]) & (time<=epoch[1])
    time = time[mask]
    signal = signal[..., mask]        

    return signal, time


def knee_freq(knee, exponent):
    """
    Convert specparam knee parameter to Hz.

    Parameters
    ----------
    knee, exponent : 1D array
        Knee and exponent parameters from specparam.

    Returns
    -------
    knee_hz : 1D array
        Knee in Hz.
    """
    
    # check if input is float or array
    if isinstance(knee, float):
        knee_hz = knee**(1/exponent)

    else:
        knee_hz = np.zeros_like(knee)
        for ii in range(len(knee)):
            knee_hz[ii] = knee[ii]**(1/exponent[ii])
        
    return knee_hz


def compute_tfr(lfp, fs, freqs, freq_spacing='lin', time_window_length=0.5, 
                freq_bandwidth=4, n_jobs=-1, decim=1, output='power', 
                verbose=False):
    """
    Compute time-frequency representation (TFR) of LFP data.

    Parameters
    ----------
    lfp : 3d array
        LFP data (trials x channels x samples).
    fs : int
        Sampling frequency.
    freqs : 1d array
        Frequency vector (start, stop, n_freqs).
    """

    # imports
    from mne.time_frequency import tfr_array_multitaper

    # define hyperparameters
    if freq_spacing == 'lin':
        freq = np.linspace(*freqs)
    elif freq_spacing == 'log':
        freq = np.logspace(*np.log10(freqs[:2]), freqs[2])
    n_cycles = freq * time_window_length # set n_cycles based on fixed time window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_array_multitaper(lfp, sfreq=fs, freqs=freq, n_cycles=n_cycles, 
                                time_bandwidth=time_bandwidth, output=output, 
                                n_jobs=n_jobs, decim=decim, verbose=verbose)

    return tfr, freq
