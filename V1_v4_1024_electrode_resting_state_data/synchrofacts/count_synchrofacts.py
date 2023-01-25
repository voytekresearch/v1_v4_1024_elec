"""Count synchrofacts

Count synchronous events in sequences of threshold crossing events.

Usage:
    count_synchrofacts.py --crossings=LIST \
                          --odml=FILE \
                          --lowSNR=FILE \
                          --lowFR=FILE \
                          --outcountstot=FILE \
                          --outcountsel=FILE \
                          --outsurrtot=FILE \
                          --outsurrel=FILE \
                          --outobj=FILE

Options:
    -h --help            Show this screen and terminate script.
    --crossings=LIST     List of paths to threshold crossing data.
    --odml=FILE          Path to meatada file.
    --lowSNR=FILE        File with list of SNR < thrsh electrode IDs
    --lowFR=FILE         File with list of FR < thrsh electrode IDs
    --outcountstot=FILE  Path to output synchronous event counts.
    --outcountsel=FILE   Path to output synchronous event counts.
    --outsurrtot=FILE    Path to output total surrogate syncounts.
    --outsurrel=FILE     Path to output electrode-wise surrogate syncounts.
    --outobj=FILE        Path to output epoch
"""
from docopt import docopt
import os
import pickle
import numpy as np
import odml
import gc
from utils import get_syncounts, get_surrogate_syncounts
import elephant

if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    crossings_paths = [path for path in vargs['--crossings'].split(';')]
    odml_path = vargs['--odml']
    metadata = odml.load(odml_path)
    lowSNR_path = vargs['--lowSNR']
    lowFR_path = vargs['--lowFR']
    counts_tot_path = vargs['--outcountstot']
    counts_el_path = vargs['--outcountsel']
    surr_tot_path = vargs['--outsurrtot']
    surr_el_path = vargs['--outsurrel']
    epoch_path = vargs['--outobj']

    # Main algorithm starts here
    sts = []
    stop = 0
    for path in crossings_paths:
        # Read threshold crossings
        with open(path, 'rb') as f:
            sts_nsp = pickle.load(f)
        # Remove waveforms to save memory
        for st in sts_nsp:
            st.waveforms = None
            if st.t_stop > stop or stop == 0:
                stop = st.t_stop
        sts += sts_nsp

    recording = metadata.sections['Recording']
    SNR_thr = recording.properties['SNR_threshold'].values[0]

    # Annotate with all metadata and correct t_stop
    sts_clean = []
    lowSNR_IDs = []
    lowFR_IDs = []
    for st in sts:
        st.t_stop = stop
        FR = elephant.statistics.mean_firing_rate(st)
        a_id = st.annotations['array_id']
        e_id = format(int(st.annotations['elec_id']), '04d')
        arr_meta = metadata.sections['Arrays'].sections[f'Array_{a_id}']
        elec_meta = arr_meta.sections[f'Electrode_{e_id}']
        propdict = {prop.name: prop.values[0] for prop in elec_meta.properties}
        st.annotations = {}
        st.annotate(**propdict)
        # Use only channels with SNR >= 2 and FR >= 0.1 crossings/s
        if st.annotations['SNR'] >= SNR_thr and FR.magnitude >= 0.1:
            sts_clean.append(st)
        # Append to one of the lists of not considered electrodes
        elif st.annotations['SNR'] < SNR_thr:
            lowSNR_IDs.append(e_id)
        elif FR.magnitude < 0.1:
            lowFR_IDs.append(e_id)
    sts = sts_clean

    # Save low FR and SNR IDs
    for path, ids in zip([lowSNR_path, lowFR_path], [lowSNR_IDs, lowFR_IDs]):
        if os.path.isfile(path):
            os.remove(path)
        with open(path, 'w') as f:
            for item in ids:
                f.writelines(str(item) + '\n')

    # Get the time histogram
    print('\tProcessing original data...')
    tot_syncounts, elec_syncounts, bins = get_syncounts(sts, path=epoch_path)
    np.save(counts_tot_path, tot_syncounts)
    np.save(counts_el_path, elec_syncounts)
    del tot_syncounts, elec_syncounts
    gc.collect()

    print(f'\tCalculating the complexity from surrogates...')
    surr_tot_arr, surr_el_arr = get_surrogate_syncounts(sts, bins)
    np.save(surr_tot_path, surr_tot_arr)
    np.save(surr_el_path, surr_el_arr)
