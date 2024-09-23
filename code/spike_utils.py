"""Spike and related simulation utility functions"""

# Imports
import numpy as np


def sample_spikes(rand_p, fs):
    """
    Sample spikes from a random process

    Parameters
    ----------
    rand_p : 1D array, float
        random process (from which spikes will be sampled).
    fs : float
        sampling frequency (1/dt).

    Returns
    -------
    spikes : 1D array, int
        spike train

    """
    # initialize
    spikes = np.zeros([len(rand_p)])
        
    # loop through each time bin
    for i_bin in range(len(rand_p)):
        # sample spikes
        if rand_p[i_bin] / fs > np.random.uniform():
            spikes[i_bin] = 1    
    
    return spikes


def sample_pop_spikes(rand_p, fs, n_neurons=10):
    """
    Simulate a population of correlated neurons

    Parameters
    ----------
    rand_p : 1D array, float
        random process (from which spikes will be sampled).
    fs : float
        sampling frequency (1/dt).
    n_neurons : int, optional
        Number of neurons in population. The default is 10.

    Returns
    -------
    spikes : 2D array, int
        population spike trains.

    """    
    
    # sample spikes
    spikes = np.zeros([n_neurons, len(rand_p)])
    for i_neuron in range(n_neurons):
        spikes[i_neuron] = sample_spikes(rand_p, fs)

    return spikes


def get_spike_times(spikes, time):
    """
    convert spike train to spike times 

    Parameters
    ----------
    spikes : 1D array, int
        Spike train.
    time : 1D array, float
        Time-vector.

    Returns
    -------
    spike_times : 1D array, float
        spike times.

    """
    
    spike_times = time[np.where(spikes)]
    
    return spike_times


def convolve_psps(spikes, fs, tau_r=0., tau_d=0.01, t_ker=None):
    """Adapted from neurodsp.sim.aperiodic.sim_synaptic_current
    
    Convolve spike train and synaptic kernel.

    Parameters
    ----------
    spikes : 1D array, int 
        spike train 
    tau_r : float, optional, default: 0.
        Rise time of synaptic kernel, in seconds.
    tau_d : float, optional, default: 0.01
        Decay time of synaptic kernel, in seconds.
    t_ker : float, optional
        Length of time of the simulated synaptic kernel, in seconds.

    Returns
    -------
    sig : 1d array
        Simulated synaptic current.
    time : 1d array
        associated time-vector (sig is trimmed  during convolution).

    """
    from neurodsp.sim.transients import sim_synaptic_kernel
    from neurodsp.utils import create_times
    
    # If not provided, compute t_ker as a function of decay time constant
    if t_ker is None:
        t_ker = 5. * tau_d

    # Simulate
    ker = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    sig = np.convolve(spikes, ker, 'valid')

    # compute time vector (convolve will trim when 'valid')
    times = create_times(len(spikes)/fs, fs)
    trim = len(times) - len(spikes)
    time = times[int(trim/2):-int(trim/2)-1]
    
    return sig, time
