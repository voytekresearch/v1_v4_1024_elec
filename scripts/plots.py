"""
Plotting functions 

"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Matplotlib params
rcParams['savefig.dpi'] = 300
rcParams['figure.facecolor'] = 'w'

def plot_psd_diff(freq, psd_diff, fname_out=None):
    """ 
    Plot spectra (or change in spectral power) in semi-log space.
    The mean spectrum is plotted in black, and the individual spectra are plotted in grey.
    A horizontal line at power=0 is also plotted. If fname_out is not None, the figure is 
    saved to that path.

    Parameters
    ----------
    freq : array
        Frequency values.
    psd_diff : array
        Spectral power values.
    fname_out : str, optional
        Path to save figure to. If None, figure is not saved.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    
    """

   # Matplotlib params
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16

    # plot psd
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(freq, psd_diff.T, color='grey')

    # plot mean
    ax.plot(freq, psd_diff.mean(axis=0), color='k', linewidth=3)

    # label
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (uV^2/Hz)')
    ax.set_title(f"Power spectrum difference")

    # annotate power=0
    ax.axhline(0, color='r', linestyle='--', linewidth=3)

    # scale x-axis logarithmically
    ax.set(xscale="log");

    # Savefig
    if not fname_out is None:
        plt.savefig(fname_out, transparent=False)

    return fig, ax


def plot_schematic(data, odml_path, fname_out=None):
    """
    Plot data from all electrodes in a schematic view.

    Adapted from:
    https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data/src/master/code/python_scripts/plotting/arrayplot_SNR.py 

    Parameters
    ----------
    data : array
        Data to plot.
    odml_path : str
        Path to odML file containing electrode metadata.
    fname_out : str, optional
        Path to save figure to. If None, figure is not saved.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects    
    
    """

    # Imports
    import odml
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.collections import PatchCollection
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

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

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
    plt.plot(samples, fun(samples), '--', c='k', lw=2)

    # Insert area labels
    plt.text(-6000, -1800, 'V4', fontsize=16)
    plt.text(-3300, -3000, 'V1', fontsize=16)

    # Formatting of plot
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    ax.set_ylim(-5400, 6000)

    # Savefig
    plt.tight_layout()
    if not fname_out is None:
        plt.savefig(fname_out, transparent=False)

    return fig, ax
