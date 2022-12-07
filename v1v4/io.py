"""
Utility import/export functions

"""

# general imports
import numpy as np

# functions
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




