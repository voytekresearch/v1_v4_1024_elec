"""Generate epoch CSV

Generate .csv files with metadata on epochs (trials or behavioral states).
RF and SNR sessions contain trials, which are stored and annotated.
RS sessions do not have any trials, eye open/closed periods are annotated.

Usage:
    generate_epochs_csv.py --ns6=FILE --out=FILE
    generate_epochs_csv.py --ns6=FILE --eyesig=FILE --out=FILE


Options:
    -h --help      Show this screen and terminate script.
    --ns6=FILE     Path to .ns6 session raw aligned file.
    --eyesig=FILE  Path to .mat eye signal path
    --out=FILE     Output file path.
"""
from docopt import docopt
import neo
import numpy as np
import pandas as pd
import quantities as pq

if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    session_path = vargs['--ns6']
    csv_path = vargs['--out']

    if 'RS' in session_path:
        raise ValueError('RS session does not have any trial to be marked!')

    # Neo IO, load events from session path
    io = neo.io.BlackrockIO(session_path)
    block = io.read_block(lazy=True)
    event_seq = block.segments[0].events[0].load()

    # Find all stimulus onsets
    stim_on = np.where(event_seq.labels == '2')[0]

    if 'RF' in session_path:

        # Check if last stim onset condition (i+2) is out of bounds
        if stim_on[-1]+2 >= len(event_seq):
            stim_on = stim_on[:-1]

        # Dict of label (8,16,32,64) to direction of sweeping bar mapping
        label_2_dir = {'8': 'rightward',
                       '16': 'upward',
                       '32': 'leftward',
                       '64': 'downward'}

        # Initialize epochs dataframe and temporary series
        epochs = pd.DataFrame(columns=['t_stim_on', 'success',
                                       't_rew', 'cond'])
        epoch_tmp = pd.Series(index=['t_stim_on', 'success', 't_rew', 'cond'],
                              dtype='float64')

        for i in stim_on:
            epoch_tmp = {}
            epoch_tmp['t_stim_on'] = event_seq[i].magnitude

            # Label mask: check if reward event comes after stim_onset
            mask_label = (event_seq.labels[i+1] == '4')

            # Timescale mask: test that condition event (i+2)
            # happens within 1.3 second range after stim_onset
            t_on = event_seq[i+2]
            t_cond = event_seq[i]
            mask_timescale = (t_cond - t_on <= 1.3*pq.s)

            # Use both masks to determine successful epochs
            if mask_label & mask_timescale:
                epoch_tmp['success'] = True
                epoch_tmp['t_rew'] = event_seq[i+1].magnitude
                epoch_tmp['cond'] = label_2_dir[event_seq.labels[i+2]]
            else:
                epoch_tmp['success'] = False
                epoch_tmp['t_rew'] = None
                epoch_tmp['cond'] = None

            epochs = epochs.append(epoch_tmp, ignore_index=True)

    elif 'SNR' in session_path:
        # Check if last stim onset reward event (i+1) is out of bounds
        if stim_on[-1]+1 >= len(event_seq):
            stim_on = stim_on[:-1]

        # Initialize epochs dataframe and temporary series
        epochs = pd.DataFrame(columns=['t_stim_on', 'success', 't_rew'])
        epoch_tmp = pd.Series(index=['t_stim_on', 'success', 't_rew'])

        for i in stim_on:
            epoch_tmp = {}
            epoch_tmp['t_stim_on'] = event_seq[i].magnitude

            # Check if reward event comes after stim_onset
            if event_seq.labels[i+1] == '4':
                epoch_tmp['success'] = True
                epoch_tmp['t_rew'] = event_seq[i+1].magnitude
            else:
                epoch_tmp['success'] = False
                epoch_tmp['t_rew'] = None

            epochs = epochs.append(epoch_tmp, ignore_index=True)

    # Save the epochs as csv
    epochs.to_csv(csv_path, index=False)
