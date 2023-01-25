"""
Plotting functions 

"""

# imports
import numpy as np
import matplotlib.pyplot as plt

def plot_schematic(data, odml_path, fname_out=None):
    """
    Plot data from all electrodes in a schematic view.

    Adapted from:
    https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data/src/master/code/python_scripts/plotting/arrayplot_SNR.py 

    """

    # Imports
    import odml
    from matplotlib import rcParams
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.collections import PatchCollection
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import LogFormatter
    from scipy.interpolate import interp1d

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
    norm = LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data), clip=True)
    cmap = plt.get_cmap('Greens', 10000)
    newcolors = cmap(np.linspace(0, 1, 10000))[2000:, :]
    cmap = ListedColormap(newcolors)

    # Iterate over electrodes to extract relevant data and create rectangles
    boxes = []
    for elec in electrodes:
        # Create rectangle objects
        x = elec.properties['schematic_X_position'].values[0] - dist/2
        y = elec.properties['schematic_Y_position'].values[0] - dist/2
        theta = float(elec.properties['schematic_rotation'].values[0])
        rect = Rectangle((x, y),
                            width=dist,
                            height=dist,
                            angle=theta)
        
        # Append to list
        boxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(boxes, norm=norm, cmap=cmap,
                         edgecolor='w', linewidth=0.2)
    pc.set_array(data)

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

    # Savefig
    plt.tight_layout()
    if not fname_out is None:
        plt.savefig(fname_out, dpi=600, transparent=True)
