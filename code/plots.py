"""
Plotting functions 

"""

# imports
import numpy as np
import matplotlib.pyplot as plt


def plot_psd_diff(freq, psd_diff, shade_conf=True, plot_each=False,
                  ax=None, fname_out=None):
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
    shade_conf : bool, optional
        Whether to shade the confidence interval of the spectra. The default is 
        True.
    plot_each : bool, optional
        Whether to plot individual spectra. The default is False.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.
    fname_out : str, optional
        Path to save figure to. If None, figure is not saved.

    Returns
    -------
    None.
    
    """

    # check for (and create) axis
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=[10,4])

    # plot mean spectra
    ax.plot(freq, psd_diff.mean(axis=0), color='k', linewidth=3)

    # plot individual spectra
    if plot_each:
        ax.plot(freq, psd_diff.T, color='grey')

    # shade between confidence interval of spectra
    if shade_conf:
        conf = np.std(psd_diff, axis=0) #/ np.sqrt(psd_diff.shape[0])
        lower_limit = np.mean(psd_diff, axis=0) - conf
        upper_limit = np.mean(psd_diff, axis=0) + conf
        ax.fill_between(freq, lower_limit, upper_limit, color='k', 
                        alpha=0.5)

    # label
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (uV^2/Hz)')
    ax.set_title(f"Power spectrum difference")

    # annotate power=0
    ax.axhline(0, color='r', linestyle='--', linewidth=3)

    # scale x-axis logarithmically
    ax.set(xscale="log")

    # Savefig
    if not fname_out is None:
        plt.savefig(fname_out, transparent=False)


def plot_schematic(data, odml_path, label=None, title=None, fname_out=None,
                   norm_type='linear', vmin=None, vmax=None, ax=None):
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
    label : str
        colorbar label.
    title : str
        figure title.
    fname_out : str, optional
        Path to save figure to. If None, figure is not saved.
    norm_type : str, optional
        Normalization type. Must be 'linear', 'log', 'centered', or 'two_slope'.
    vmin : float, optional
        Minimum value for colorbar. If None, minimum value of data is used.
    vmax : float, optional
        Maximum value for colorbar. If None, maximum value of data is used.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.

    Returns
    -------
    None.
    
    """

    # Imports
    import odml
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, LogNorm, CenteredNorm, TwoSlopeNorm
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

    # Create figure and axes
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Define a color map and normalization of values
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    if norm_type == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'centered':
        norm = CenteredNorm(vcenter=0)
        cmap = 'coolwarm'
    elif norm_type == 'two_slope':
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap = 'coolwarm'
    else:
        print("norm_type must be 'linear', 'log', 'centered', or 'two_slope'")

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
    ax.plot(samples, fun(samples), '--', c='k', lw=2)

    # Insert area labels
    ax.text(-6000, -1800, 'V4', fontsize=16)
    ax.text(-3300, -3000, 'V1', fontsize=16)

    # Formatting of plot
    ax.axis('equal')
    ax.axis('off')
    # ax.tight_layout()
    ax.set_ylim(-5400, 6000)

    # Colorbar
    axin1 = inset_axes(ax,
                       width="100%", height="100%",
                       bbox_to_anchor=(0.15, 0.15, 0.3, 0.03),
                       bbox_transform=ax.transAxes)
    if norm_type == 'log':
        formatter = LogFormatter(10, labelOnlyBase=False)
        cb = plt.colorbar(pc,
                        cax=axin1,
                        orientation='horizontal',
                        format=formatter,
                        extend='max'
                        )
    else:
        cb = plt.colorbar(pc,
                        cax=axin1,
                        orientation='horizontal')
    if not label is None:
        cb.ax.set_xlabel(label)

    # add title
    if not title is None:
        ax.set_title(title)

    # Savefig
    # plt.tight_layout()
    if not fname_out is None:
        plt.savefig(fname_out, transparent=False)


def plot_schematic_var(df, odml_path, var, y_label, dir_figs):
    """
    Plot multiple schematics for a list of variables

    Parameters
    ----------
    df : dataFrame
    odml_path : str
        Path to odML file containing electrode metadata.
    var : str
      variable of interest
    y_label : str
      label for y-axis
    dir_figs : str
      directory to save figures
    """

    # get list of sessions and loop through
    sessions = df['session'].unique()
    for i_session, session in enumerate(sessions):
        # display progress
        print(f"    plotting session ({i_session+1}/{len(sessions)}): {session}")

        # get data for session
        df_session = df[df['session']==session]

        # setting limits for colorbar
        lims = np.append(df_session[f"{var}_pre"], df_session[f"{var}_post"])
        maxli = np.nanmax(lims)
        minli = np.nanmin(lims)
        
        # pre-stimulus
        plot_schematic(df_session[f"{var}_pre"], odml_path, title=f"session: {session}", 
                       label=f"pre-stimulus {y_label}", vmin=minli, vmax=maxli, 
                       fname_out=f"{dir_figs}/{session}_{var}_pre.png");

        # post-stimulus
        plot_schematic(df_session[f"{var}_post"], odml_path, title=f"session: {session}", 
                       label=f"post-stimulus {y_label}", vmin=minli, vmax=maxli, 
                       fname_out=f"{dir_figs}/{session}_{var}_post.png");

        # change in exponent
        diff = df_session[f"{var}_post"] - df_session[f"{var}_pre"]
        plot_schematic(diff, odml_path, norm_type="centered", title=f"session: {session}", 
                       label=f"difference in {y_label}", vmin=np.nanmin(diff), vmax=np.nanmax(diff), 
                       fname_out=f"{dir_figs}/{session}_{var}_diff.png");

        plt.close('all')


def plot_spectra_2conditions(spectra_a, spectra_b, freq, ax=None, shade_sem=True,
                             color=['k','b'], labels=['baseline','encoding'],
                             y_units='\u03BCV\u00b2/Hz'):
    
    """
    Plot mean spectra for two conditions, with optional shading of SEM.

    Parameters
    ----------
    spectra_a : 2d array
        Power spectra for condition a.
    spectra_b : 2d array
        Power spectra for condition b.
    freq : 1d array
        Frequency values corresponding to spectra.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.
    shade_sem : bool, optional
        Whether to shade SEM. The default is True.
    color : list, optional
        Colors for each condition. The default is ['k','b'].
    labels : list, optional
        Labels for each condition. The default is ['baseline','encoding'].

    Returns
    -------
    None.
    """

    # check axis
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=[6,4])

    # check psds are 2d
    if not (spectra_a.ndim == 2 and spectra_b.ndim == 2):
        raise ValueError('PSDs must be 2d arrays.')

    # plot mean spectra for each condition
    ax.loglog(freq, np.nanmean(spectra_a, axis=0), color=color[0], 
              label=labels[0])
    ax.loglog(freq, np.nanmean(spectra_b, axis=0), color=color[1], 
              label=labels[1]) 
    
    # shade between SEM of spectra for each condition
    if shade_sem:
        for spectra, col in zip([spectra_a, spectra_b], color):
            sem = np.std(spectra, axis=0) / np.sqrt(spectra.shape[0])
            lower_limit = np.mean(spectra, axis=0) - sem
            upper_limit = np.mean(spectra, axis=0) + sem
            ax.fill_between(freq, lower_limit, upper_limit, color=col, 
                            alpha=0.5)

    # set axes ticks and labels
    ax.set_ylabel(f'power ({y_units})')
    ax.set_xlabel('frequency (Hz)')
    ax.set_xticks([10, 100])
    ax.set_xticklabels(["10", "100"])

    # add legend
    ax.legend(labels)


def plot_erp(erp, time, x_units='s', y_units='\u03BCV', 
    annotate_times=[0], legend_labels=None, ax=None):
    """Plots a voltage versus time graph of an evoked response (ERP).

    Parameters
    ----------
    erp : array_like
        Voltage vector of the ERP.
    time : array_like
        Time vector of the ERP.
    x_units : str, optional
        Units for the x-axis (default is 's').
    y_units : str, optional
        Units for the y-axis (default is 'μV').
    annotate_times : list, optional
        Time points to annotate with a vertical line (default is [0]).
    legend_labels : list, optional
        List of labels for the legend (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on (default is None).

    Returns
    -------
    None.

    """

    # create figure
    if ax is None:
        fig, ax = plt.subplots()

    # one ERP only
    if np.ndim(erp) == 1:
        # plot
        ax.plot(time, erp)

    # multiple ERPs
    elif np.ndim(erp) ==2:
        # plot
        ax.plot(time, erp.T)

        # label
        if not legend_labels is None:
            ax.legend(legend_labels)

    # label
    ax.set(xlabel=f'time ({x_units})', ylabel=f'voltage ({y_units})')
    ax.set_title('Evoked Response')

    # annotate
    if annotate_times is not None:
        for annotate_time in annotate_times:
            ax.axvline(annotate_time, color='k', linestyle='--')

    else:
        raise ValueError('ERP must be a 1D or 2D array')


def plot_event_traces(event_traces, time, annotate_time=0,
    plot_mean=True, plot_std=True, plot_individual=False,
    x_units='s', y_units='\u03BCV', ax=None):

    """Plots event traces and related statistics.

    Parameters
    ----------
    event_traces : array_like
        An array of event traces
    time : array_like
        An array of time points corresponding to the event traces
    annotate_time : int, optional
        The time point at which to draw an annotation line. The default is 0
    plot_mean : bool, optional
        Whether or not to plot the mean trace. The default is True
    plot_std : bool, optional
        Whether or not to plot the standard deviation. The default is True
    plot_individual : bool, optional
        Whether or not to plot individual traces. The default is False
    x_units : str, optional
        The string to use in the x-axis label. The default is 's'
    y_units : str, optional
        The string to use in the y-axis label. The default is 'µV'
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. The default is None.

    Returns
    -------
    None.
    """

    # create figure
    if ax is None:
        fig, ax = plt.subplots()

    # plot individual traces
    if plot_individual:
        ax.plot(time, event_traces.T, color='grey', alpha=0.5)

    # plot mean
    et_mean = np.mean(event_traces, axis=0)
    if plot_mean:
        ax.plot(time, et_mean, color='k')

    # plot standard deviation
    if plot_std:
        et_std = np.std(event_traces, axis=0)
        ax.fill_between(time, et_mean-et_std, et_mean+et_std, color='grey', alpha=0.5)

    # label
    ax.set(xlabel=f'time ({x_units})', ylabel=f'voltage ({y_units})')
    ax.set_title('Evoked Response')

    # annotate
    if not annotate_time is None:
        ax.axvline(annotate_time, color='k', linestyle='--')


def plot_regression(x_data, y_data, ax=None, title=None, xlabel=None, 
                    ylabel=None, color=None, fname_out=None, label='', label_offset=0):
    """
    Calculate and plot the linear regression for two datasets, with optional labels and file output.
    Parameters
    ----------
    x_data : 1-d array_like
        x-values of the dataset
    y_data : 1-d array_like
        y-values of the dataset
    title : str, optional
        Title of the plot
    xlabel : str, optional
        x-axis label of the plot
    ylabel : str, optional
        y-axis label of the plot
    fname_out : str, optional
        Filename of the output figure

    Returns
    -------
    None

    """

    # imports
    from scipy.stats import linregress
    
    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        fig.patch.set_facecolor('white') # set background color to white for text legibility
        
    # plot data
    if color is not None:
        ax.scatter(x_data, y_data, c=color, alpha=0.1)
    else:
        ax.scatter(x_data, y_data, alpha=0.1)

    # run regression and plot results
    results = linregress(x_data, y_data)
    t_lin = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
    lin = results.slope * t_lin + results.intercept
    if color is not None:
        ax.plot(t_lin, lin, c=color, label=label, linewidth=3)
    else:
        ax.plot(t_lin, lin, label=label, linewidth=3)

    # add regression results text
    if results.pvalue < 0.001:
        pval = f"{results.pvalue:.2e}"
    else:
        pval = f"{results.pvalue:.3f}"
    ax.text(1.05, 0.85-label_offset, 
                f"{label}\n" +
                f"    Slope: {results.slope:.3f}\n" +
                f"    Intercept: {results.intercept:.3f}\n" +
                f"    R: {results.rvalue:.3f}\n" +
                f"    p: {pval}",
                transform=ax.transAxes)

    # label figure
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    # save/show figure
    if not fname_out is None:
        fig.savefig(fname_out)

