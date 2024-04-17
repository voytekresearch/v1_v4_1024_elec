# -*- coding: utf-8 -*-
"""
Utility functions for TFR analysis
"""

# Imports
import numpy as np


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


def zscore_tfr(tfr):
    """
    Normalize time-frequency representation (TFR) by z-scoring each frequency.
    TFR should be 2D (frequency x time).

    Parameters
    ----------
    tfr : 2D array
        Time-frequency representation of power (spectrogram).

    Returns
    -------
    tfr_norm : 2D array
        Z-score normalized TFR.
    """
    
    # initialize 
    tfr_norm = np.zeros(tfr.shape)
    
    # z-score normalize 
    for i_freq in range(tfr.shape[0]):
        tfr_norm[i_freq] = (tfr[i_freq] - np.mean(tfr[i_freq])) / np.std(tfr[i_freq])
        
    return tfr_norm


def subtract_baseline(signals, time, t_baseline=None):
    """
    Subtract baseline from signals. Baseline is defined as the mean of the
    signal between t_baseline[0] and t_baseline[1]. Signals should be 2D
    (signals x time).

    Parameters
    ----------
    signals : 2D array
        Signals to be baseline corrected.
    time : 1D array
        Time vector.
    t_baseline : 1D array
        Time range for baseline (t_start, t_stop).

    Returns
    -------
    signals_bl : 2D array
        Baseline corrected signals.
    """
    
    # initialize
    signals_bl = np.zeros_like(signals)

    # set mask for baseline time window
    if t_baseline is None:
        mask_bl = (time<0)
    else:
        mask_bl = ((time>t_baseline[0]) & (time<t_baseline[1]))    
    if sum(mask_bl)==0:
        raise ValueError('Baseline time window is empty. Check t_baseline.')
    
    # subtract baseline from each signal
    for ii in range(len(signals)):
        bl = np.mean(signals[ii, mask_bl])
        signals_bl[ii] = signals[ii] - bl
    
    return signals_bl


def trim_tfr(tfr, freq, time, freq_range=None, time_range=None):
    """
    Crop time-frequency representation (TFR) to specified time and frequency 
    ranges.

    Parameters
    ----------
    tfr : 2D array
        Time-frequency representation of power (spectrogram).
    freq : 1D array
        Associated frequency vector (length should be equal to that of
        the first dimension of tfr).
    time : 1D array
        Associated time vector (length should be equal to that of
        the last dimension of tfr).
    freq_range : 1D array
        Frequency range to keep (f_start, f_stop).
    time_range : 1D array
        Time range to keep (t_start, t_stop).

    Returns
    -------
    tfr, freq, time : arrays
        Cropped TFR, frequency-, and time-vectors.
    """

    # crop frequency
    if not freq_range is None:
        freq_mask = np.logical_and(freq>freq_range[0], freq<freq_range[1])
        tfr = tfr[freq_mask, :]
        freq = freq[freq_mask]

    # crop time
    if not time_range is None:
        time_mask = np.logical_and(time>time_range[0], time<time_range[1])
        tfr = tfr[:, time_mask]
        time = time[time_mask]
    
    return tfr, freq, time


def crop_tfr(tfr, time, time_range):
    """
    Crop time-frequency representation (TFR) to time_range.
    TFR can be mulitdimensional (time must be last dimension).

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of
        the last dimension of tfr).
    time_range : 1D array
        Time range to crop (t_start, t_stop).

    Returns
    -------
    tfr, time : array, array
        Cropped TFR and time vector.
    """
    
    tfr = tfr[..., (time>time_range[0]) & (time<time_range[1])]
    time = time[(time>time_range[0]) & (time<time_range[1])]
    
    return tfr, time


def downsample_tfr(tfr, time, n):
    """
    Downsample time-frequency representation (TFR) to n time bins.
    TFR can be mulitdimensional (time must be last dimension)

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of 
        the last dimension of tfr).
    n : int
        Desired number of time bins after downsampling.

    Returns
    ------- 
    tfr, time : array, array
        Downsampled TFR and time vector.
    """

    # determine step size for downsampling and counnt number of samples
    n_samples = len(time)
    step = int(np.floor(tfr.shape[-1]/n))

    # downsample
    tfr = tfr[..., np.arange(0, n_samples-1, step)] 
    time = time[np.arange(0, n_samples-1, step)] 
    
    return tfr, time


def preprocess_tfr(tfr, time, downsample_n=None, edge=None, average_trials=True, z_score=True, t_baseline=None):

    # downsample
    if not downsample_n is None:
        tfr, time = downsample_tfr(tfr, time, downsample_n)

    # crop edge effects
    if not edge is None:
        tfr, time = crop_tfr(tfr, time, [time[0]+edge/2,time[-1]-edge/2])

    # average spectrogram over trials
    if average_trials:
        tfr = np.nanmedian(tfr, axis=0)

    # normalize (zscore)
    if z_score:
        tfr = zscore_tfr(tfr)

    # subtract basline
    if not t_baseline is None:
        tfr = subtract_baseline(tfr, time, t_baseline)

    return tfr, time


def load_tfr_results(fname, preprocess=True, downsample_n=None, edge=None, 
                     average_trials=True, z_score=True, subtract_baseline=False,
                     t_baseline=None):
    # load data
    data_in = np.load(fname)

    # pre-process
    if preprocess:
        tfr, time = preprocess_tfr(data_in['spectrogram'], data_in['time'], 
                                   downsample_n=downsample_n,  edge=edge, 
                                   average_trials=average_trials, 
                                   z_score=z_score, 
                                   subtract_baseline=subtract_baseline,
                                   t_baseline=t_baseline)
        
    return time, data_in['freq'], tfr
