"""Array plot response timing

Plot the stimulus evoked response timing for each electrode on a
schematic diagram of the arrays.

Usage:
    arrayplot_response_timing.py --odml=FILE --out=FILE

Options:
    -h --help     Show this screen and terminate script.
    --odml=FILE   Path to .odml metadata file.
    --out=FILE    Output file path.

"""
from docopt import docopt
import odml
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, to_rgba, ListedColormap
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    odml_path = vargs['--odml']
    out_path = vargs['--out']

    # Load metadata
    metadata = odml.load(odml_path)

    # Put all electrodes in a list
    arrays = metadata['Arrays']
    electrodes = []
    for array in arrays.sections:
        for elec in array.sections:
            electrodes.append(elec)

    # Get distance between electrodes (relevant for creating rectangles)
    dist = float(arrays.properties['ElectrodeSeparation'].values[0])

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

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))

    # Define a color map and normalization of values
    norm = Normalize(vmin=0, vmax=150, clip=True)
    cmap = plt.get_cmap('Spectral', 10000)
    newcolors = cmap(np.linspace(0, 1, 10000))
    newcolors[:int(norm(20)*10000), :] = to_rgba('dimgray')
    newcolors[int(norm(140)*10000):, :] = to_rgba('dimgray')
    cmap = ListedColormap(newcolors)

    # Iterate over electrodes to extract relevant data and create rectangles
    boxes = []
    vals = []
    for elec in electrodes:
        # Extract the value of interest
        val = elec.properties['response_onset_timing'].values[0]
        vals.append(val)

        # Create rectangle objects
        x = elec.properties['schematic_X_position'].values[0] - dist/2
        y = elec.properties['schematic_Y_position'].values[0] - dist/2
        theta = float(elec.properties['schematic_rotation'].values[0])
        rect = Rectangle((x, y),
                         width=dist,
                         height=dist,
                         angle=theta)
        boxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(boxes, norm=norm, cmap=cmap,
                         edgecolor='w', linewidth=0.2)
    pc.set_array(np.array(vals))

    # Add collection to axes
    ax.add_collection(pc)

    # Draw separation line between V1 and V4
    x = [1400, -250, -2800, -4000]
    y = [4500, 2700, 720, -2600]
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

    # Add inset axes with a histogram of the SNR
    axin = inset_axes(ax,
                      width="100%", height="100%",
                      bbox_to_anchor=(0.48, 0.74, 0.42, 0.12),
                      bbox_transform=ax.transAxes)

    # Create a histogram of the values
    hist, bin_edges = np.histogram(vals,
                                   bins=np.arange(20, 140, step=5))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Create histogram as barplot
    axin.bar(bin_centers,
             hist,
             width=bin_centers[2] - bin_centers[1],
             color=cmap(norm(bin_centers)))

    # Add vertical line to mark onset
    axin.vlines(x=[0], ymin=5, ymax=1.1*np.max(hist), linewidth=0.6)
    axin.text(-2, 1.15*np.max(hist), 'STIM_ON', fontsize=6)

    # Format histogram plot
    axin.spines['right'].set_visible(False)
    axin.spines['top'].set_visible(False)
    axin.set_ylabel('Number of\n electrodes')
    axin.set_xlabel('Time\n (ms)')
    axin.xaxis.set_label_coords(1.13, 0.2)

    # Colorbar
    axin1 = inset_axes(ax,
                       width="100%", height="100%",
                       bbox_to_anchor=(0.15, 0.3, 0.3, 0.015),
                       bbox_transform=ax.transAxes)
    cb = plt.colorbar(pc,
                      cax=axin1,
                      orientation='horizontal',
                      extend='max'
                      )
    cb.ax.set_xlabel('Timing of response (ms)')

    # Savefig
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, transparent=True)
