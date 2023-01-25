"""Plot eye signal and states

Generates review plot of eye states.

Usage:
    eye_states_plot.py --eyenix=FILE --muae=FILE --out=FILE

Options:
    -h --help      Show this screen and terminate script.
    --eyenix=FILE  Path to .nix file of eye signals.
    --muae=FILE     Path to .nix file of MUAe from one array.
    --out=FILE     Output file path.

"""
from docopt import docopt
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import gridspec
import neo
import quantities as pq
import scipy

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


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    eye_path = vargs['--eyenix']
    mua_path = vargs['--muae']
    out_path = vargs['--out']

    fname = eye_path.split('/')[-3]

    # Colors
    coldict = {'Closed_eyes': "dimgray",
               'Open_eyes': "gold"}
    lineplots_witdh = 0.3
    OC_alpha = 0.5
    OC_colours = ['dimgray', 'gold']

    # Load data
    with neo.NixIO(eye_path, mode='ro') as nio:
        block = nio.read_block()

    epc = block.segments[0].epochs[0]

    xdiam = block.segments[0].analogsignals[2]
    ydiam = block.segments[0].analogsignals[3]
    diam = np.sqrt(xdiam.magnitude**2 + ydiam.magnitude**2)

    behavioural_state = block.segments[0].analogsignals[4].magnitude[:, 0]

    # Build mosaic
    fig, axs = plt.subplot_mosaic('''
                                  ..AAAAAAAAAAAA..
                                  ..AAAAAAAAAAAA..
                                  ..BBBBBBBBBBBB..
                                  ..BBBBBBBBBBBB..
                                  ..CCCCCCCCCCCC..
                                  ..CCCCCCCCCCCC..
                                  ................
                                  ................
                                  ................
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  DDDDDDDDDDEEEEEE
                                  ''', figsize=(4,3))

    fig.patch.set_facecolor('white')

    for letter in axs:
        for spine in axs[letter].spines:
            axs[letter].spines[spine].set_visible(False)

    """ Plot eye  diameter signal """
    ax = axs['A']
    ax.plot(diam, color='k', lw=lineplots_witdh)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Pupil\ndiameter', rotation=0,
                  va='center', ha='right', fontsize=7)

    """ Plot state """
    ax = axs['B']
    ax.plot(behavioural_state, color='k', lw=lineplots_witdh)
    ax.set_ylabel('Eyes open\n/ closed', rotation=0,
                  va='center', ha='right', fontsize=7)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.get_shared_x_axes().join(axs['A'], axs['B'])
    # Color code behavioural epochs
    edges = np.array([0] + list(np.where(np.diff(behavioural_state != 0))[0])
                 + [len(behavioural_state)])
    left = edges[:-1]
    widths = np.diff(edges)
    colorlist = [OC_colours[int(val)]
                 for val in behavioural_state[left+widths//2]]
    ax.barh(0.5, widths, height=1, left=left, color=colorlist, alpha=OC_alpha)
    ax.set_xticks([])
    ax.set_yticks([])

    """ Plot MUAe """
    ax = axs['C']
    # Load data
    with neo.NixIO(mua_path, mode='ro') as nio:
        block = nio.read_block()
        mua = block.segments[0].analogsignals[0]

    # Downsample
    factor = 1000  # 1 Hz at output
    dwn = mua.downsample(factor, ftype='fir')
    mua = dwn.magnitude

    # Z-score and take median
    mua = (mua - mua.mean(axis=0)) / mua.std(axis=0)
    mua = np.median(mua, axis=1)
    
    # Display
    ax.plot(np.arange(0, len(mua))[5:-5], mua[5:-5],
            color='k', lw=lineplots_witdh)
    ax.get_shared_x_axes().join(axs['C'], axs['B'])
    ax.get_shared_x_axes().join(axs['C'], axs['A'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MUAe', rotation=0, va='center', ha='right', fontsize=7)
    ax.set_yticks([])

    # Annotate testing results
    axin = ax.inset_axes([1, 0.3, 0.1, 1.5])
    if '140819' in eye_path:
        r2, p = scipy.stats.pearsonr(mua[72:], behavioural_state)
    else:
        r2, p = scipy.stats.pearsonr(mua, behavioural_state)
    axin.plot([0.1, 0.2, 0.2, 0.1], [0, 0, 1, 1], c='k', lw=1)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_xlim(0, 1)
    for s in axin.spines:
        axin.spines[s].set_visible(False)

    if p < 0.01:
        axin.text(0.35, 0.5, 'Pearson:\n'
                  r'R$^2$ = ' + '{:.2f}'.format(r2) +
                  '\n' + 'p = ' + '{:.0e}'.format(p),
                  fontsize=7, va='center')
    else:
        axin.text(0.35, 0.5, 'Pearson:\n'
                  r'R$^2$ = ' + '{:.2f}'.format(r2) +
                  '\n' + 'p = ' + '{:.2f}'.format(p),
                  fontsize=7, va='center')
    ax.spines['bottom'].set_visible(True)

    """ Total duration of each state """
    ax = axs['D']

    axin = ax.inset_axes([0.1, 0, 0.75, 1])
    p_open = np.sum(behavioural_state)/len(behavioural_state)
    p_closed = 1 - p_open
    x = [p_closed, p_open]
    percentages = ['{:.2f}%'.format(p*100) for p in x]
    if 'L' in eye_path: angle = -45
    if 'A' in eye_path: angle = -75
    axin.pie(x, colors=OC_colours,
             startangle=angle,
             labels=percentages,
             labeldistance=1.15,
             wedgeprops=dict(width=0.5, edgecolor='w', alpha=OC_alpha),
             textprops={'fontsize': 6})
    axin.set_xlabel('Total time in\n each state')
    ax.set_xticks([])
    ax.set_yticks([])

    # Make histogram of epoch length
    ax = axs['E']
    off = 0
    xtickpos = [0]
    xticklbl = [0]
    coldict = {'Closed_eyes': 'dimgray',
               'Open_eyes': 'gold'}
    for lbl in np.unique(epc.labels):
        # Draw bars only for periods longer than 100 ms
        mask = (epc.labels == lbl) & (epc.durations > 100*pq.ms)
        durations = -np.sort(-epc.durations[mask])
        num = np.arange(len(durations))
        pos = num + off
        ax.bar(pos,
               durations.magnitude,
               color=coldict[lbl],
               alpha=OC_alpha,
               linewidth=0.1)
        # Should set three ticks in total
        step = int(len(pos)/3)
        xtickpos += pos[:1:-step].tolist()
        xticklbl += (num + 1)[:1:-step].tolist()
        off += int(1.1*len(durations))

    # gridlines
    ax.set_yscale('log')
    ax.set_xticks(xtickpos)
    ax.set_xticklabels(xticklbl)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Duration (s)')
    ax.set_xlabel('Segments ordered by duration')

    axin = ax.inset_axes([0.5, 0.95, 0.1, 0.3])
    # Make legend for behavioural epochs
    legend_elements = [Line2D([0], [0], marker='.', color='none',
                              markeredgecolor='none', label=f'Eyes open',
                              markerfacecolor=OC_colours[1],
                              alpha=OC_alpha, markersize=13),
                       Line2D([0], [0], marker='.', color='none',
                              markeredgecolor='none', label=f'Eyes closed',
                              markerfacecolor=OC_colours[0],
                              alpha=OC_alpha, markersize=13)]
    axin.axis('off')
    axin.legend(handles=legend_elements, fontsize=7, ncol=1,
                handletextpad=-0.3, frameon=False,
                loc='center left').set_zorder(1000)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('pdf', 'png'), dpi=300, bbox_inches='tight')
