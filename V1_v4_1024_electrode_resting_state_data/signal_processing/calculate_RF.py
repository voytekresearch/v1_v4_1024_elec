"""Calculate RF

Calculate the Receptive Field (RF) of each electrode from the multiunit
activity envelope (MUAe).


Usage:
    calculate_RF.py --muae=FILE --odml=FILE --out=FILE --plt=DIR

Options:
    -h --help     Show this screen and terminate script.
    --muae=FILE   MUAe signal from SNR session.
    --odml=FILE   Path to .odml metadata file.
    --plt=FILE    File template where to put helper RF response plots
    --out=FILE    Output file path.

"""
from docopt import docopt
import os
import odml
import neo
import neo.utils
import quantities as pq
import numpy as np
import scipy
import pandas as pd
from utils import odml2quantity
from copy import copy
import matplotlib.pyplot as plt


def gaussian(x, amplitude, mu, sigma):
    return np.abs(amplitude) * np.exp(- (((x-mu)**2)/(2*sigma**2))) # [CK]


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    muae_path = vargs['--muae']
    odml_path = vargs['--odml']
    plt_path = vargs['--plt']
    out_path = vargs['--out']

    # Load odml
    metadata = odml.load(odml_path)

    # Convenience indexing of sections in odml
    stim = metadata['Recording']['Trials']['Stimulus']

    # Get stimulus duration information
    stim_dur = odml2quantity(stim.properties['Stimulus_duration'])[0]
    pre_dur = odml2quantity(stim.properties['Pre_stim_duration'])[0]

    pixperdeg = odml2quantity(stim.properties['Pixels_per_degree'])[0]
    x0 = stim.properties['bar_x0'].values[0]
    y0 = stim.properties['bar_y0'].values[0]
    speed = odml2quantity(stim.properties['bar_speed'])[0]

    bardist = speed*stim_dur

    # Different condition names
    directions = stim.properties['Directions'].values
    stim.properties['Direction_angles'].unit = 'deg'
    angles = pq.Quantity(np.array(stim.properties['Direction_angles'].values),
                         stim.properties['Direction_angles'].unit)
    SNR_thresh = metadata['Recording'].properties['SNR_threshold'].values[0]
    r2_thresh = 0.4  # hard coded

    # Load the data
    with neo.NixIO(muae_path, mode='ro') as io:
        block = io.read_block()
    segment = block.segments[0]

    # Extract other relevant metadata from segment
    sampling_rate = segment.analogsignals[0].sampling_rate
    array_annot = segment.analogsignals[0].array_annotations

    # Initialize storage place
    response_start = {}
    response_stop = {}
    fit_goodness = {}
    MUA_sm_all = []
    SNR = {}

    # Iterate over the different bar directions
    for epc in segment.epochs:

        # Allocate empty list to each entry
        name = epc.name
        response_start[name] = []
        response_stop[name] = []
        fit_goodness[name] = []

        # Extract trials into a list of segments
        seg_lst = neo.utils.cut_segment_by_epoch(copy(segment),
                                                 epc, reset_time=True)

        # Create a list of neo.AnalogSignal
        MUA_lst = []
        for seg in seg_lst:
            anasig = seg.analogsignals[0].magnitude
            MUA_lst.append(anasig)

        # Turn list into a 3D numpy.ndarray
        MUA_stack = np.stack(MUA_lst, axis=-1)

        # Get noise levels before smoothing
        baseline_index = int((sampling_rate*pre_dur.rescale(pq.s)).magnitude)
        baseline_stack = MUA_stack[0:baseline_index]
        baseline_stack_avg = np.mean(baseline_stack, axis=0)
        baseline_avg = np.mean(baseline_stack_avg, axis=-1)
        baseline_std = np.std(baseline_stack_avg, axis=-1)

        # Take mean across all trials in same channel
        MUA_avg = np.mean(MUA_stack, axis=-1)

        # Smoothen MUA with a moving average (via convolution)
        w = 20  # hard coded
        mask = np.ones((w)) / w
        MUA_sm = scipy.ndimage.convolve1d(MUA_avg, mask, axis=0)
        MUA_sm_all.append(MUA_sm)

        # Calculate SNR
        MUA_max = np.max(MUA_sm, axis=0)
        SNR[name] = (MUA_max - baseline_avg) / baseline_std

        # Prepare for gaussian fit
        MUA_fit = ((MUA_sm - baseline_avg) / baseline_std).T

        # Define span of the signal
        s0, s1 = MUA_fit.shape
        unit = pq.ms
        dur = epc.durations[0].rescale(unit).magnitude
        indices = np.array([np.linspace(0, dur-1, num=s1)]*s0)
        indices -= pre_dur.rescale(unit).magnitude  # Align to stimulus onset

        # Establish initial guesses
        amplitude_guesses = np.max(MUA_fit, axis=1)
        mu_guesses = np.argmax(MUA_fit, axis=1)
        sigma_guesses = 500*np.ones(s0)
        initial_guesses = np.array([amplitude_guesses,
                                    mu_guesses,
                                    sigma_guesses]).T

        # Estimate a gaussian curve and get response on/off times
        # Creates four plots of the corresponding response
        fig, axs = plt.subplots(8, 8, figsize=(16, 16))
        for x, y, guess, snr, ax in zip(indices, MUA_fit, initial_guesses,
                                        SNR[name], axs.flat):
            on = np.nan
            off = np.nan
            r2 = np.nan
            if snr >= SNR_thresh:
                try:
                    # Estimate curve and goodness of fit
                    maxiter = 50000
                    curve, _ = scipy.optimize.curve_fit(gaussian, x, y,
                                                    p0=guess,
                                                    bounds=((0,guess[1]-400,0),(np.inf, guess[1]+400,np.inf)),
                                                    maxfev=maxiter)
                    # residual sum of squares
                    ss_res = np.sum((y - gaussian(x, *curve)) ** 2)
                    # total sum of squares
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    # r-squared
                    r2 = 1 - (ss_res / ss_tot)

                    if r2 >= r2_thresh:
                        # Estimate onset and offset
                        on = curve[1] - 1.65*curve[2]
                        off = curve[1] + 1.65*curve[2]

                        # Generate plot for visual inspection
                        ax.plot(x, y, color='b', alpha=0.5)
                        ax.plot(x, gaussian(x, *curve), color='k')
                        ax.text(0.95, 0.9, f'$R^2 = {str(r2)[:4]}$',
                                ha='right', va='center',
                                transform=ax.transAxes)
                    else:
                        # if the goodness of fit is under the threshold
                        ax.plot(x, y, color='gray', alpha=0.2)
                        ax.plot(x, gaussian(x, *curve), color='gray',
                                alpha=0.4)
                        ax.text(0.5, 0.5, f'$R^2 < {r2_thresh}$',
                                ha='center', va='center',
                                transform=ax.transAxes)
                    # [CK: print fit results for debugging]
                    print('Guess (m,sd) = ' + guess[1].astype('str') + ', '
                          + guess[2].astype('str') +
                          '; Fit (m,sd) = ' + curve[1].astype('str') + ', '
                          + curve[2].astype('str') +
                          '; R2 = ' + r2.astype('str')  )
                except RuntimeError:
                    # If the fit does not converge
                    ax.plot(x, y, color='gray', alpha=0.2)
                    ax.text(0.5, 0.5, 'Gaussian fit\n did not converge',
                            ha='center', va='center', transform=ax.transAxes)
            else:
                # if SNR does not go over the threshold
                ax.plot(x, y, color='gray', alpha=0.2)
                ax.text(0.5, 0.5, f'SNR < {SNR_thresh}',
                        ha='center', va='center', transform=ax.transAxes)
            response_start[name].append(on*unit)
            response_stop[name].append(off*unit)
            fit_goodness[name].append(r2)
        fig.text(0.5, 0.05, 'Time relative to stimulus onset (ms)',
                 ha='center', fontsize=16)
        fig.text(0.05, 0.5, 'Trial average z-scored MUAe',
                 va='center', rotation='vertical', fontsize=16)
        fig.suptitle(f"{plt_path.split('/')[-1]}_{name}, SNR > {SNR_thresh}",
                     fontsize=24)
        dirplot_path = f'{plt_path}_{name}.png'
        dirpath = os.path.dirname(dirplot_path)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        plt.savefig(dirplot_path, dpi=300)
        plt.clf()

    # Get starting position of bars (in pixels)
    sx = x0 + bardist * np.cos(angles.rescale(pq.rad)) / 2
    sy = y0 + bardist * np.sin(angles.rescale(pq.rad)) / 2

    # Shape the ons and off into a more useful format
    ons = []
    offs = []
    for dir in directions:
        ons.append(response_start[dir])
        offs.append(response_stop[dir])
    ons = np.array(ons).T * unit
    offs = np.array(offs).T * unit

    # For each channel
    RF_edges = {}
    RF_edges['top'] = []
    RF_edges['bottom'] = []
    RF_edges['left'] = []
    RF_edges['right'] = []
    for on, off in zip(ons, offs):
        # Distance travelled by the bars in pixels
        ondist = speed * on
        offdist = speed * off

        # Angular distance moved
        # (direction is opposite to angle of start position)
        on_angx = ondist * np.cos((180*pq.deg-angles).rescale(pq.rad))
        on_angy = ondist * np.sin(angles.rescale(pq.rad))
        off_angx = offdist * np.cos((180*pq.deg-angles).rescale(pq.rad))
        off_angy = offdist * np.sin(angles.rescale(pq.rad))

        # on and off points are starting position + angular distance
        onx = sx + on_angx
        ony = sy - on_angy
        offx = sx + off_angx
        offy = sy - off_angy

        # RF boundaries
        bottom = (ony[1] + offy[3]).simplified.magnitude / 2
        right = (onx[2] + offx[0]).simplified.magnitude / 2
        top = (ony[3] + offy[1]).simplified.magnitude / 2
        left = (onx[0] + offx[2]).simplified.magnitude / 2

        # Save RF data
        RF_edges['top'].append(top)
        RF_edges['bottom'].append(bottom)
        RF_edges['left'].append(left)
        RF_edges['right'].append(right)

    date = metadata['Recording'].properties['Date'].values[0]

    # Put into a pandas dataframe
    data_dict = {'Electrode_ID': array_annot['Electrode_ID'],
                 'Array_ID': array_annot['Array_ID'],
                 'date': date.strftime("%Y-%m-%d")
                 }

    for d in directions:
        data_dict['SNR_fromRF_' + d] = np.array(SNR[d])
        data_dict['R2_' + d] = np.array(fit_goodness[d])
        data_dict['response_onset_' + d] = np.array(response_start[d])
        data_dict['response_offset_' + d] = np.array(response_stop[d])

    for d in ['top', 'bottom', 'left', 'right']:
        data_dict['RF_' + d + '_edge (pixels)'] = np.array(RF_edges[d])

    # RF centers
    centrex = (data_dict['RF_right_edge (pixels)'] +
               data_dict['RF_left_edge (pixels)']) / 2
    centrey = (data_dict['RF_top_edge (pixels)'] +
               data_dict['RF_bottom_edge (pixels)']) / 2

    # RF characteristics
    x = centrex / pixperdeg
    y = centrey / pixperdeg
    data_dict['RF center X (degrees)'] = x
    data_dict['RF center Y (degrees)'] = y
    data_dict['RF theta (degrees)'] = np.arctan2(y, x) * 180 / np.pi

    r_l = (data_dict['RF_right_edge (pixels)'] -
           data_dict['RF_left_edge (pixels)']) / pixperdeg
    t_b = (data_dict['RF_top_edge (pixels)'] -
           data_dict['RF_bottom_edge (pixels)']) / pixperdeg
    data_dict['RF size (degrees)'] = np.sqrt(r_l**2 + t_b**2)

    # Save the metadata as csv
    df = pd.DataFrame(data=data_dict)
    df.to_csv(out_path, index=False)
