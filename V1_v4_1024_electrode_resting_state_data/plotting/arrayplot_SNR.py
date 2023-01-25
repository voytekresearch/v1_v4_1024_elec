"""Array plot SNR

Plots the Signal to noise ratio (SNR) on a schematic representation
of the arrays.


Usage:
    arrayplot_SNR.py --odml=FILE --out=FILE

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
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogFormatter
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

    # Get SNR threshold
    SNR_thresh = metadata['Recording'].properties['SNR_threshold'].values[0]

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
    norm = LogNorm(vmin=2, vmax=35, clip=True)
    cmap = plt.get_cmap('Greens', 10000)
    newcolors = cmap(np.linspace(0, 1, 10000))[2000:, :]
    cmap = ListedColormap(newcolors)

    # Iterate over electrodes to extract relevant data and create rectangles
    boxes = []
    vals = []
    for elec in electrodes:
        # Extract the value of interest
        val = elec.properties['SNR'].values[0]

        # Create rectangle objects
        x = elec.properties['schematic_X_position'].values[0] - dist/2
        y = elec.properties['schematic_Y_position'].values[0] - dist/2
        theta = float(elec.properties['schematic_rotation'].values[0])

        if val < 2:
            ax.text(x+np.cos((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    y+np.sin((theta+45)*np.pi/180)*dist/np.sqrt(2),
                    'x', ha='center', va='center',
                    rotation=theta, fontsize=4)
        else:
            rect = Rectangle((x, y),
                             width=dist,
                             height=dist,
                             angle=theta)
            boxes.append(rect)
            vals.append(val)

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
                                   bins=np.arange(0, 35, step=1))
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
    axin.set_xlabel('SNR')
    axin.xaxis.set_label_coords(1.1, 0.1)

    # Pie plot with portion of good channels
    axinin = inset_axes(axin,
                        width="100%", height="100%",
                        bbox_to_anchor=(0.8, 0.5, 0.15, 0.75),
                        bbox_transform=axin.transAxes)
    pievals = [len(vals), 1024-len(vals)]
    axinin.pie(pievals, labels=pievals,
               colors=[cmap(norm(10)), 'gray'],
               wedgeprops=dict(width=0.6, edgecolor='w', linewidth=0.5))
    axinin.set_xlabel(r'SNR $\geq$ 2')
    axinin.xaxis.set_label_coords(0.5, 1.35)

    # Colorbar
    axin1 = inset_axes(ax,
                       width="100%", height="100%",
                       bbox_to_anchor=(0.15, 0.3, 0.3, 0.015),
                       bbox_transform=ax.transAxes)
    formatter = LogFormatter(10, labelOnlyBase=False)
    cb = plt.colorbar(pc,
                      cax=axin1,
                      orientation='horizontal',
                      format=formatter,
                      extend='max'
                      )
    cb.ax.set_xlabel('SNR')

    # Savefig
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, transparent=True)
