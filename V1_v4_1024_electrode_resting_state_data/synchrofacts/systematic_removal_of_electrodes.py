"""Systematic removal of electrodes

Removes electrodes from session aiming to minimize cross talk

Usage:
    systematic_removal_of_electrodes.py --ses=STR \
                                        --odml=FILE \
                                        --lowFR=FILE \
                                        --syn-obj=FILE \
                                        --orig-tot-cpx=FILE \
                                        --orig-el-cpx=FILE \
                                        --surr-tot-cpx=FILE \
                                        --surr-el-cpx=FILE \
                                        --plot-dir=DIR \
                                        --out-csv=FILE

Options:
    -h --help            Show this screen and terminate script.
    --ses=STR            Session identifier
    --odml=FILE          Path to metadata file.
    --lowFR=FILE         File with list of FR < thrsh electrode IDs
    --syn-obj=FILE       Path to synchrotool object
    --orig-tot-cpx=FILE  Path to original total complexity histogram
    --orig-el-cpx=FILE   Path to original electrode-wise complexity histogram
    --surr-tot-cpx=FILE  Path to surrogate total complexity histogram
    --surr-el-cpx=FILE   Path to surrogate electrode-wise complexity histogram
    --plot-dir=DIR       Directory to store output plots
    --out-csv=FILE       Path to output csv of the removal process
"""
import pickle
import odml
import numpy as np
import pandas as pd
import os
import gc
from os.path import join
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import DivergingNorm, ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from docopt import docopt
from utils import get_syncounts, get_surrogate_syncounts

# Matplotlib params
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['axes.labelsize'] = 7
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['xtick.major.size'] = 2
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.size'] = 2
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.major.pad'] = '2'
rcParams['ytick.major.pad'] = '2'
rcParams['axes.linewidth'] = 0.5

# Define a color map and normalization of values for spatial plots
norm = DivergingNorm(vmin=-0, vcenter=0.08, vmax=1)
cmap = plt.get_cmap('coolwarm')
# The midpoint is at 0.05 after removing the darkest blue hues
newcolors = cmap(np.linspace(0, 1, 1000))[300:, :]
cmap = ListedColormap(newcolors)

# Colormap for p values
G_cmap = plt.get_cmap('Greens_r')
O_cmap = plt.get_cmap('Oranges_r')
orangecolors = O_cmap(np.linspace(0, 1, 500))[250:, :]
greencolors = G_cmap(np.linspace(0, 1, 2200))
newcolors = np.concatenate((orangecolors[:11, :], greencolors[251:1251, :]))
p_cmap = ListedColormap(newcolors)


def get_excess_and_pvalues(tot_hist, surr_tot_hist, el_hist, surr_el_hist):
    # Calculate p values of the total complexity distribution
    tot_pvals = []
    for i, val in enumerate(tot_hist):
        if val == 0:
            tot_pvals.append(np.nan)
        elif i >= surr_tot_hist.shape[-1]:
            tot_pvals.append(0)
        else:
            surrogate_dist = np.sort(surr_tot_hist[:, i])
            x = np.searchsorted(surrogate_dist, val)
            S = len(surrogate_dist) - x
            N = len(surrogate_dist)
            pval = S / N
            tot_pvals.append(pval)
    tot_pvals = np.array(tot_pvals)

    # Calculate p values and excess complexity electrode-wise
    el_pvals = np.zeros(el_hist.shape)
    el_pvals[:, :] = np.nan
    el_excess = np.zeros(el_hist.shape)
    el_excess[:, :] = np.nan
    for i, occ in enumerate(el_hist):
        for j, val in enumerate(occ):
            if val != 0:
                surrogate_dist = np.sort(surr_el_hist[i, j, :])
                x = np.searchsorted(surrogate_dist, val)
                if i >= surr_el_hist.shape[0]:
                    el_pvals[i, j] = 0
                    el_excess[i, j] = val - surrogate_dist[x-1]
                else:
                    S = len(surrogate_dist) - x
                    N = len(surrogate_dist)
                    pval = S / N
                    el_pvals[i, j] = pval
                    if pval <= 0.01:  # hard coded
                        el_excess[i, j] = val - surrogate_dist[x-1]
    excess_sum = np.nansum(el_excess, axis=0)

    return excess_sum, tot_pvals, el_pvals


def draw_complexities(tot_cpxhist, surr_tot_cpxhist, tot_pvals, el_cpxhist,
                      el_pvals, SP, pdfpath):
    # Draw complexity plots

    mean_tot_chance = np.mean(surr_tot_cpxhist, axis=0)
    std_tot_chance = np.std(surr_tot_cpxhist, axis=0)

    # Plotting the excess complexity
    fig, axs = plt.subplots(2, 2, figsize=(4, 6),
                            gridspec_kw={'width_ratios': [4, 1]})

    # Total cpxhist
    ax = axs[0, 0]
    sizes_c = np.arange(1, len(mean_tot_chance)+1, step=1)
    ax.bar(np.arange(1, len(tot_pvals)+1), tot_cpxhist,
           color=p_cmap(tot_pvals), linewidth=0)
    ax.plot(sizes_c, mean_tot_chance, c='k',
            label='Mean chance level', linewidth=0.5)
    ax.fill_between(sizes_c, mean_tot_chance+2*std_tot_chance,
                    mean_tot_chance-2*std_tot_chance, alpha=0.5,
                    facecolor='k',
                    zorder=1000, label=r'$\pm$ 2std chance level')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlim(0.5, len(sizes_c)+0.5)
    ax.set_ylim(0.8, np.max(tot_cpxhist)*5)
    ax.set_yscale('log')
    ax.set_ylabel('Occurrences')
    ax.legend()
    # pvalue colorbar
    axin = inset_axes(ax,
                      width="100%", height="100%",
                      bbox_to_anchor=(0.25, 0.75, 0.7, 0.02),
                      bbox_transform=ax.transAxes)
    sm = ScalarMappable(cmap=p_cmap, norm=plt.Normalize(0, 1))
    cb = plt.colorbar(sm,
                      cax=axin,
                      orientation='horizontal')
    cb.ax.set_xlabel('p-value')

    # Unused axis
    axs[0, 1].axis('off')

    # Breakdown by electrodes
    ax = axs[1, 0]
    ordered_idxs = np.flip(np.argsort(SP))
    ax.imshow(el_pvals[:, ordered_idxs].T, cmap=p_cmap, aspect='auto',
              extent=[0.5, el_pvals.shape[0] + 0.5, 0, el_pvals.shape[1]])
    ax.set_ylim(0, el_cpxhist.shape[1])
    ax.set_xlim(0.5, len(sizes_c)+0.5)
    ax.set_ylabel('Electrodes')
    ax.set_xlabel('Complexity')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Excess per electrode summed
    ax = axs[1, 1]
    ax._shared_y_axes.join(ax, axs[1, 0])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.barh(np.flip(range(len(ordered_idxs))), SP[ordered_idxs],
            color=cmap(norm(SP[ordered_idxs])))
    ax.set_ylim(0, el_cpxhist.shape[1])
    ax.set_xlabel('SP')

    plt.tight_layout()
    plt.savefig(pdfpath)
    plt.clf()
    plt.close()


def draw_arrayplot(metadata, SP, low_FRs, elec_lbls, removed_elecs, plotpath):
    # Put all electrodes in a list
    arrays = metadata['Arrays']
    electrodes = []
    for array in arrays.sections:
        for elec in array.sections:
            electrodes.append(elec)

    # Get distance between electrodes, relevant for creating rectangles
    dist = float(arrays.properties['ElectrodeSeparation'].values[0])

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))

    # Iterate over electrodes to extract relevant data
    boxes = []
    vals = []
    snrs = []
    elecs = []
    for elec in electrodes:
        # Extract the value of interest
        elec_id = elec.properties['Electrode_ID'].values[0]
        elecs.append(elec_id)
        snr = elec.properties['SNR'].values[0]
        snrs.append(snr)

        # Create rectangle objects
        x = elec.properties['schematic_X_position'].values[0] - dist/2
        y = elec.properties['schematic_Y_position'].values[0] - dist/2
        theta = float(elec.properties['schematic_rotation'].values[0])

        if snr < 2:
            ax.text(x+np.cos((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    y+np.sin((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    'x', ha='center', va='center',
                    rotation=theta, fontsize=4)
        elif elec_id in removed_elecs:
            ax.text(x+np.cos((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    y+np.sin((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    'o', ha='center', va='center',
                    rotation=theta, fontsize=4, color='darkgreen')
        else:
            rect = Rectangle((x, y),
                             width=dist,
                             height=dist,
                             angle=theta)
            boxes.append(rect)
            val = SP[np.where(elec_lbls == elec_id)].astype(float)
            if len(val) == 0:
                val = 0
            elif elec_id in low_FRs:
                val = 0
            else:
                val = val[0]
            vals.append(val)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(boxes, norm=norm, cmap=cmap,
                         edgecolor='w', linewidth=0.2)
    pc.set_array(np.array(vals))

    # Add collection to axes
    ax.add_collection(pc)

    # Draw separation line between V1 and V4
    x = [1400, -250, -2800, -4000]
    y = [4500, 2700, 720, -2300]
    fun = interp1d(x, y, kind='cubic')
    samples = np.linspace(max(x), min(x), num=100)
    plt.plot(samples, fun(samples), '--', c='k', lw=0.8)

    # Insert area labels
    plt.text(-6000, -1800, 'V4', fontsize=9)
    plt.text(-3300, -3000, 'V1', fontsize=9)

    # Formatting of plot
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    ax.set_ylim(-5400, 6000)

    # Add inset axes with a histogram
    axin = inset_axes(ax,
                      width="100%", height="100%",
                      bbox_to_anchor=(0.48, 0.74, 0.42, 0.12),
                      bbox_transform=ax.transAxes)

    # Create a histogram of the values
    hist, bin_edges = np.histogram(vals,
                                   bins=np.arange(0, 1, step=0.01))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Create histogram as barplot
    axin.bar(bin_centers,
             hist,
             width=bin_centers[1] - bin_centers[0],
             color=cmap(norm(bin_centers)))

    # Format histogram plot
    axin.spines['right'].set_visible(False)
    axin.spines['top'].set_visible(False)
    axin.set_ylabel('Number of\n electrodes')
    axin.set_xlabel('SP')
    axin.set_yscale('log')
    axin.xaxis.set_label_coords(1.08, 0.1)

    axin1 = inset_axes(ax,
                       width="100%", height="100%",
                       bbox_to_anchor=(0.17, 0.3, 0.3, 0.015),
                       bbox_transform=ax.transAxes)

    # Colorbar
    cb = plt.colorbar(pc, cax=axin1, orientation='horizontal')
    cb.ax.set_xlabel('Synchrofact Participation (SP)')
    cb.set_clim(0, 1)

    # Create legend for markers
    if "A_RS" in plotpath:
        shift = 5800
    else:
        shift = 0
    ax.text(9500+shift, -4800, ' x ', color='k', fontsize=6,
            bbox=dict(facecolor='none', edgecolor='k',
                      linewidth=0.2, boxstyle='round,pad=0.1'))
    ax.text(10800+shift, -4800, "SNR < 2", fontsize=6)
    ax.text(9500+shift, -6000, ' o ', color='darkgreen', fontsize=6,
            bbox=dict(facecolor='none', edgecolor='k',
                      linewidth=0.2, boxstyle='round,pad=0.1'))
    ax.text(10800+shift, -6000, "High SP", fontsize=6)

    # Savefig
    plt.savefig(plotpath, dpi=600)
    plt.clf()
    plt.close()


def draw_removal_process(highest_SP_lst, highest_size_lst, rmv_path):
    """ Plot removal process """
    fig, host = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()

    p1, = host.plot(range(len(highest_SP_lst)), highest_SP_lst, color='k')
    p2, = par1.plot(range(len(highest_size_lst)), highest_size_lst, "red")

    host.set_xlabel('Number of removed electrodes')
    host.set_ylabel('Highest SP of any electrode')
    par1.set_ylabel('Size of largest synchrofact')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    # Set limits of right hand yaxis
    maxsize = max(highest_size_lst)
    par1.set_ylim(0, maxsize+0.5)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    plt.title('Systematic removal of electrodes')
    plt.tight_layout()
    plt.savefig(rmv_path)
    plt.clf()
    plt.close()


if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    ses = vargs['--ses']
    odml_path = vargs['--odml']
    low_FRs_path = vargs['--lowFR']
    synobj_path = vargs['--syn-obj']
    tot_path = vargs['--orig-tot-cpx']
    el_path = vargs['--orig-el-cpx']
    surr_tot_path = vargs['--surr-tot-cpx']
    surr_el_path = vargs['--surr-el-cpx']
    plotdir = vargs['--plot-dir']
    out_csv = vargs['--out-csv']

    # Load metadata
    metadata = odml.load(odml_path)
    with open(low_FRs_path, 'r') as f:
        low_FRs = np.array(f.read().split()).astype(int)

    # Load the spiketrains from the synchrotool object
    with open(synobj_path, 'rb') as f:
        synobj = pickle.load(f)
        sts = synobj.input_spiketrains.copy()
        del synobj
        gc.collect()

    # Load original data complexity histograms (for initialization)
    tot_cpxhist = np.load(tot_path)
    el_cpxhist = np.load(el_path)

    # Load surrogate complexity histograms (for initialization)
    surr_tot_cpxhist = np.load(surr_tot_path)
    surr_el_cpxhist = np.load(surr_el_path)

    highest_SP_lst = []
    highest_size_lst = []
    removed_elecs = []
    # Procedure will remove up to 250 electrodes
    discard_nums = 250
    for iteration in range(discard_nums):

        # Calculate excess and p-values of the complexities
        excess_sum, tot_pvals, el_pvals = \
            get_excess_and_pvalues(tot_cpxhist, surr_tot_cpxhist,
                                   el_cpxhist, surr_el_cpxhist)

        # Get SP from excess and number of threshold crossings
        N = np.array([len(st) for st in sts])
        FRs = np.array([len(st) for st in sts])
        SP = excess_sum / N

        # Make summary plot of complexities
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)
        pdfpath = join(plotdir, f'cpxhistograms_{ses}_minus{iteration}.pdf')
        draw_complexities(tot_cpxhist, surr_tot_cpxhist, tot_pvals,
                          el_cpxhist,  el_pvals, SP, pdfpath)

        # Exit removal procedure if no synchrofacts are found
        if np.max(SP) == 0:
            highest_SP_lst.append(0)
            highest_size_lst.append(len(tot_cpxhist))
            removed_elecs.append(None)
            outdict = {'Iteration': np.arange(len(removed_elecs)).tolist(),
                       'Highest SP': highest_SP_lst,
                       'Removed electrode ID': removed_elecs,
                       'Largest complexity': highest_size_lst}
            df = pd.DataFrame(data=outdict)
            df.to_csv(out_csv, index=False)
            break

        # Get electrode ids from remaining spiketrains
        elec_ids = np.array([st.annotations['Electrode_ID'] for st in sts])

        # Array plot of SP and removal process
        arrplt_path = join(plotdir, f'SP_plot_{ses}_minus_{iteration}.pdf')
        draw_arrayplot(metadata, SP, low_FRs, elec_ids, removed_elecs, arrplt_path)

        # Save the progress into lists
        highest_SP_lst.append(np.max(SP))
        highest_size_lst.append(len(tot_cpxhist))

        """ Remove electrode from original data """
        # Remove highest SP electrode
        highest_SP_idx = np.where(SP == np.max(SP))[0][0]
        removed_elec_id = elec_ids[highest_SP_idx]
        removed_elecs.append(removed_elec_id)
        st = sts.pop(highest_SP_idx)
        print(f'\tRemoved electrode {removed_elec_id}')

        """ Recalculate complexity histograms of original data"""
        tot_cpxhist, el_cpxhist, bins = get_syncounts(sts)
        np.save(tot_path.replace('.npy', f"_minus{iteration+1}.npy"),
                tot_cpxhist)
        np.save(el_path.replace('.npy', f"_minus{iteration+1}.npy"),
                el_cpxhist)

        """ Recalculate surrogate data """
        surr_tot_cpxhist, surr_el_cpxhist = get_surrogate_syncounts(sts, bins)
        np.save(surr_tot_path.replace('.npy', f"_minus{iteration+1}.npy"),
                surr_tot_cpxhist)
        np.save(surr_el_path.replace('.npy', f"_minus{iteration+1}.npy"),
                surr_el_cpxhist)

        """ Saving removal progress """
        # Save the metadata as csv
        outdict = {'Iteration': np.arange(len(removed_elecs)).tolist(),
                   'Highest SP': highest_SP_lst,
                   'Removed electrode ID': removed_elecs,
                   'Largest complexity': highest_size_lst}
        df = pd.DataFrame(data=outdict)
        df.to_csv(out_csv, index=False)

        # Draw removal process
        rmv_path = join(plotdir, f'removal_progress_{ses}.pdf')
        draw_removal_process(highest_SP_lst, highest_size_lst, rmv_path)
