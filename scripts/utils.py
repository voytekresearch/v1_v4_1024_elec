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


def epoch_neo_segment(segment, reset_time=True):
    """Split a Neo segment into epochs and reshape to 3D array.

    Parameters
    ----------
    segment : neo.Segment
        Neo segment object to be epoched. Must contain epochs.
    reset_time : bool, optional
        Whether to reset the time of the epochs to 0. The default is True.
    Returns
    -------
    signal_epoch : np.ndarray
        Matrix of epochs with shape (trials x channels x time).
    """

    # imports
    from neo.utils import cut_segment_by_epoch
    from copy import copy

    # use Neo utils to epoch
    seg_lst = cut_segment_by_epoch(copy(segment), segment.epochs[0], reset_time=reset_time)

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
    Epochs a segment from a NIX file.

    Parameters
    ----------
    fname : str
        File name of the NIX file.

    Returns
    -------
    signal_epoch: array
        Array of epochs of shape (trials x channels x time).
    """

    # imports
    from io import load_nix
    
    # Load nix
    segment, _ = load_nix(fname)

    # epoch
    signal_epoch = epoch_neo_segment(segment)

    return signal_epoch