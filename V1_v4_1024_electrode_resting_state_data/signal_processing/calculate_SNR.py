"""Calculate SNR

Calculate the signal to noise ratio (SNR) per channel from the multiunit
activity envelope (MUAe).


Usage:
    calculate_SNR.py --muae=FILE --odml=FILE --plt=FILE --out=FILE

Options:
    -h --help     Show this screen and terminate script.
    --muae=FILE   MUAe signal from SNR session.
    --odml=FILE   Path to .odml metadata file.
    --plt=FILE    File template where to put helper RF response plots
    --out=FILE    Output file path.

"""
from docopt import docopt
import odml
import neo
import neo.utils
import quantities as pq
import numpy as np
import scipy
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    muae_path = vargs['--muae']
    odml_path = vargs['--odml']
    plt_path = vargs['--plt']
    out_path = vargs['--out']

    # Load odml
    metadata = odml.load(odml_path)

    # Get stimulus duration information
    trials = metadata['Recording']['Trials']
    stim = trials['Stimulus']
    stim_dur = pq.Quantity(stim.properties['Stimulus_duration'].values[0],
                           stim.properties['Stimulus_duration'].unit)
    pre_dur = pq.Quantity(stim.properties['Pre_stim_duration'].values[0],
                          stim.properties['Pre_stim_duration'].unit)
    post_dur = pq.Quantity(stim.properties['Post_stim_duration'].values[0],
                           stim.properties['Post_stim_duration'].unit)
    SNR_thresh = metadata['Recording'].properties['SNR_threshold'].values[0]

    # Load the data
    with neo.NixIO(muae_path, mode='ro') as io:
        block = io.read_block()
        seg = block.segments[0]
        del block

    # Extract other relevant metadata from segment
    sampling_rate = seg.analogsignals[0].sampling_rate
    array_annot = seg.analogsignals[0].array_annotations
    epoch = seg.epochs[0]

    # Extract trials into a list of segments
    seg_lst = neo.utils.cut_segment_by_epoch(copy(seg), epoch, reset_time=True)
    del seg

    # Create a list of analogsignal objects
    MUA_lst = []
    for seg in seg_lst:
        anasig = seg.analogsignals[0].rescale(pq.uV).magnitude
        MUA_lst.append(anasig)

    # Turn list into a 3D numpy.ndarray
    MUA_stack = np.stack(MUA_lst, axis=-1)

    # Get noise levels before smoothing
    baseline_index = int((sampling_rate*pre_dur.rescale(pq.s)).magnitude)
    baseline_stack = MUA_stack[0:baseline_index]
    baseline_stack_avg = np.mean(baseline_stack, axis=0)
    baseline_avg = np.mean(baseline_stack_avg, axis=-1)
    baseline_std = np.std(baseline_stack_avg, axis=-1)

    # Take mean across all trials for each channel
    MUA_avg = np.mean(MUA_stack, axis=-1)

    # Smoothen MUA with a moving average (via convolution)
    window = 20  # hard-coded
    mask = np.ones((window)) / window
    MUA_sm = scipy.ndimage.convolve1d(MUA_avg, mask, axis=0)
    MUA_max = np.max(MUA_sm, axis=0)

    # Calculate channel Signal to Noise Ratio (SNR)
    SNR = (MUA_max - baseline_avg) / baseline_std

    # Create plots for SNR signals
    s0, s1 = MUA_avg.shape
    unit = pq.ms
    dur = epoch.durations[0].rescale(unit).magnitude
    indices = np.array([np.linspace(0, dur-1, num=s0)]*s1)
    indices -= pre_dur.rescale(unit).magnitude  # Align to stimulus onset
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    Y = ((MUA_avg - baseline_avg) / baseline_std).T
    YM = ((MUA_sm - baseline_avg) / baseline_std).T
    for x, y, ym, snr, ax in zip(indices, Y, YM, SNR, axs.flat):
        if snr >= SNR_thresh:
            # Generate plot for visual inspection
            ax.plot(x, y, color='b', alpha=0.5)
            ax.plot(x, ym, color='k')
            ax.text(0.95, 0.9, f'$SNR = {str(snr)[:4]}$',
                    ha='right', va='center',
                    transform=ax.transAxes)
        else:
            # if SNR does not go over the threshold
            ax.plot(x, y, color='gray', alpha=0.2)
            ax.plot(x, ym, color='darkgray', alpha=0.5)
            ax.text(0.5, 0.5, f'SNR = {str(snr)[:4]}',
                    ha='center', va='center', transform=ax.transAxes)
    fig.text(0.5, 0.05, 'Time relative to stimulus onset (ms)',
             ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'Trial average z-scored MUAe',
             va='center', rotation='vertical', fontsize=16)
    name = plt_path.split('/')[-1].split('.png')[0]
    fig.suptitle(f"{name}, SNR > {SNR_thresh}", fontsize=24)
    plt.savefig(plt_path, dpi=300)
    plt.clf()

    # Calculate response delay: time when the signal is 1 std over baseline
    SNR_thresh = metadata['Recording'].properties['SNR_threshold'].values[0]
    stds_over_baseline = SNR_thresh
    thresh = baseline_avg + stds_over_baseline*baseline_std
    response_bool = (MUA_avg[baseline_index:] > thresh).astype('float')

    # Only when it stays over baseline for some consecutive bins
    consecutive_bins = 5  # hard-coded
    zeros = np.zeros((consecutive_bins-1))
    ones = np.ones((consecutive_bins)) / consecutive_bins
    mask = np.concatenate((ones, zeros))
    # The order of the mask array is flipped by scipy
    response_bool_con = scipy.ndimage.convolve1d(response_bool, mask, axis=0,
                                                 mode='constant', cval=0)
    response_index = np.argmax(response_bool_con >= 1, axis=0)

    # Give the adequate units
    response_delay = (response_index / sampling_rate).rescale(pq.ms)

    date = metadata['Recording'].properties['Date'].values[0]

    # Put into a pandas dataframe
    dict = {'Electrode_ID': array_annot['Electrode_ID'],
            'date': date.strftime("%Y-%m-%d"),
            'SNR': SNR,
            'peak_response (uV)': MUA_max,
            'baseline_avg (uV)': baseline_avg,
            'baseline_std (uV)': baseline_std,
            'response_onset_timing (ms)': response_delay.magnitude}
    df = pd.DataFrame(data=dict)

    # Save the metadata as csv
    df.to_csv(out_path, index=False)
