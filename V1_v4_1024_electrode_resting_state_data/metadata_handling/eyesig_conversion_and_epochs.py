"""Eye signal to nix format

1. Convert the eye signals into the nix format;
2. extract whether eyes are open/closed to a csv file; and
3. annotate the analog signal object with an epoch.

Usage:
    eyesig_conversion_and_epochs.py --eyesig=FILE --csv=FILE --out=FILE

Options:
    -h --help      Show this screen and terminate script.
    --eyesig=FILE  Path to .mat file of eye signals.
    --csv=FILE     Path to .csv file of eye signals.
    --out=FILE     Output file path.

"""
from docopt import docopt
import scipy.io
import neo
import quantities as pq
from datetime import datetime as date
from neo.io.nixio import NixIO
import numpy as np
import pandas as pd
import yaml
import os
import pathlib
import elephant
from elephant.signal_processing import butter

# Read parameters directly from the configfile
dirname = pathlib.Path(__file__).absolute().parent.parent
with open(os.path.join(dirname, 'configfile.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    # Sampling rate
    sampling_rate = pq.Quantity(config['samplerate'],
                                config['samplerate_unit'])

    units = config['eyesig_unit']


def get_mean_state(diam):
    if np.sum(diam <= 0.5) > np.sum(diam > 0.5):
        state = 'Closed_eyes'
    else:
        state = 'Open_eyes'
    return state


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    eye_path = vargs['--eyesig']
    csv_path = vargs['--csv']
    out_path = vargs['--out']

    # Load eye signals
    mat = scipy.io.loadmat(eye_path)

    # Get all the analogsignals
    xpos = mat['eyeData']['XPos'][0][0][0]
    ypos = mat['eyeData']['YPos'][0][0][0]
    xdiam = mat['eyeData']['XDiam'][0][0][0]
    if 'A_' in eye_path:
        xdiam = xdiam/10
    ydiam = mat['eyeData']['YDiam'][0][0][0]
    ydiam = ydiam - np.min(ydiam)
    xdiam = xdiam - np.min(xdiam)

    signals = [xpos, ypos, xdiam, ydiam]
    names = ['XPos', 'YPos', 'XDiam', 'YDiam']

    # Create block and segment objects to hold the data
    fullBlock = neo.Block(name='Eye signals',
                            description='eye position and diameters',
                            file_origin=eye_path,
                            file_datetime=date.now())
    fullSegment = neo.Segment(name='eye signal segment',
                                description="""Segment of eye pos and diam""",
                                file_origin=eye_path)
    fullSegment.analogsignals = []

    # Turn eye signals into neo analogsignals
    for sig, name in zip(signals, names):
        anasig = neo.core.AnalogSignal(sig,
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       name=name)
        fullSegment.analogsignals.append(anasig)

    """ Downsample eye signals """
    # Original rate was 230Hz
    # Downsampling to 1Hz is done filter out blinking
    downsample_factor = 30000
    smallBlock = neo.Block(name='Downsampled eye signals',
                           description='Downsampled eye position and diameters',
                           file_origin=eye_path,
                           file_datetime=date.now())
    smallSegment = neo.Segment(name='Downsampled eye signals',
                               description="""Segment of downsampled eye
                                              diameter and position""",
                               file_origin=eye_path)
    for sig in fullSegment.analogsignals:
        small_signal = sig.downsample(downsample_factor, ftype='fir')
        smallSegment.analogsignals.append(small_signal)

    # Pre-process signals in downsampled file
    xpos = smallSegment.analogsignals[0] / 10000
    ypos = smallSegment.analogsignals[1] / 10000
    xdiam = smallSegment.analogsignals[2] / 1000
    ydiam = smallSegment.analogsignals[3] / 1000
    xdiam[xdiam < 0] = 0*pq.mV
    ydiam[ydiam < 0] = 0*pq.mV
    if 'A_' in eye_path:
        # Manual processing for the Aston signals
        xpos = xpos*2000
        xpos = xpos - xpos.mean()
        if '150819' in eye_path:
            # Removes low frequency reference change
            xpos = butter(xpos, highpass_freq=0.05*pq.Hz)

    smallSegment.analogsignals[0] = xpos
    smallSegment.analogsignals[1] = ypos
    smallSegment.analogsignals[2] = xdiam
    smallSegment.analogsignals[3] = ydiam

    """ Estimate behavioural epochs """
    # Use both x and y diameter signals
    diam = np.sqrt(ydiam.magnitude**2 + xdiam.magnitude**2)

    # Empirically estimated thresholds for eye closure
    if 'L_RS_250717' in eye_path: thr = 0.0
    if 'L_RS_090817' in eye_path: thr = 0.101
    if 'L_RS_100817' in eye_path: thr = 0.058
    if 'A_RS_140819' in eye_path: thr = 0.34
    if 'A_RS_150819' in eye_path: thr = 0.47
    if 'A_RS_160819' in eye_path: thr = 0.831

    # Mask of behavioual state
    mask = (diam > thr)
    behavioural_state = mask.astype(float)[:, 0]
    behaviour_anasig = neo.core.AnalogSignal(behavioural_state,
                                             units=pq.V,
                                             sampling_rate=1*pq.Hz,
                                             name='Behavioural state')
    smallSegment.analogsignals.append(behaviour_anasig)

    # Smoothen states with sliding window
    w = 3
    kernel = [1/w]*w
    behavioural_state = np.convolve(behavioural_state, kernel, mode='same')
    behavioural_state[behavioural_state < 0.5] = 0
    behavioural_state[behavioural_state >= 0.5] = 1

    """ Create epoch object """
    wh = np.where(np.diff(behavioural_state) != 0)[0]
    edgeindex = [0] + wh.tolist() + [len(behavioural_state)]

    # initialise with first slice
    i_start = [edgeindex[0]]
    i_stop = [edgeindex[1]]
    states = [get_mean_state(behavioural_state[edgeindex[0]:edgeindex[1]])]
    # Loop over indices, assign states and merge consecutive same-state slices
    for startidx, stopidx in zip(edgeindex[1:-1], edgeindex[2:]):
        nextstate = get_mean_state(behavioural_state[startidx:stopidx])
        if nextstate == states[-1]:
            i_stop[-1] = stopidx
        else:
            i_start.append(startidx)
            i_stop.append(stopidx)
            states.append(nextstate)

    # Turn index lists into time arrays
    start_times = (np.array(i_start) / ydiam.sampling_rate).rescale('s')
    stop_times = (np.array(i_stop) / ydiam.sampling_rate).rescale('s')
    durs = stop_times - start_times

    # Convert into a pandas dataframe,
    datadict = {'t_start': start_times.magnitude,
                't_stop': stop_times.magnitude,
                'dur': durs.magnitude,
                'state': states}

    epochs = pd.DataFrame(data=datadict)

    # Save the epochs as csv
    epochs.to_csv(csv_path, index=False)

    # Create the epoch object
    epc = neo.core.Epoch(times=start_times,
                         durations=durs,
                         labels=states)

    # Save fullscale data to .nix
    fullBlock.segments = [fullSegment]
    fullBlock.segments[0].epochs.append(epc)
    with NixIO(out_path, mode='ow') as nio:
        nio.write(fullBlock)

    # Save downsampled signals to .nix
    smallBlock.segments = [smallSegment]
    smallBlock.segments[0].epochs.append(epc)
    small_path =out_path.replace('.nix', '_downsampled_1Hz.nix')
    with NixIO(small_path, mode='ow') as nio:
        nio.write(smallBlock)
