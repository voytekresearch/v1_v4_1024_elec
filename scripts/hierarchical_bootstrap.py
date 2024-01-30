# imports
import numpy as np


def prep_data(df, feature, condition, levels):
    """
    Prepares data for hierarchical bootstrap.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data. Columns must include:
            - feature
            - condition
            - levels
    feature : str
        Feature of interest. These values will be compared between conditions.
    condition : str
        Condition of interest. Must have exactly two conditions.
    levels : list
        List of hierarchical levels. For example, if the data is organized by
        subject, neuron, and trial, then levels = ['subject', 'neuron', 'trial'].

    Returns
    -------
    data_0, data_1 : array
        Data arrays to be compared. Each dimension of the array represents a
        hierarchical level. For example, the first dimension could represent
        subjects, the seconds dimension could represent neurons, and the third
        dimension could represent trials.
    
    """
    # seperate conditions of interest (must be two)
    conditions = df[condition].unique()
    if len(conditions) != 2:
        raise ValueError('More/less than two conditions detected. Please check your data.')
    df_0 = df[df[condition] == conditions[0]]
    df_1 = df[df[condition] == conditions[1]]

    # get instances of each level
    level_instances = dict()
    for i, level in enumerate(levels):
        level_instances[level] = df_0[level].unique()

    # initialize each output array
    data_0 = np.zeros(tuple([len(df_0[level].unique()) for level in levels]))
    data_1 = np.zeros(tuple([len(df_1[level].unique()) for level in levels]))

    # fill arrays with feature values
    for row_0, row_1 in zip(df_0.itertuples(), df_1.itertuples()):
        idx_0 = tuple([np.where(level_instances[level] == getattr(row_0, level))[0][0] for i, level in enumerate(levels)])
        idx_1 = tuple([np.where(level_instances[level] == getattr(row_1, level))[0][0] for i, level in enumerate(levels)])
        data_0[idx_0] = getattr(row_0, feature)
        data_1[idx_1] = getattr(row_1, feature)

    return data_0, data_1


def hierarchical_bootstrap(data_0, data_1, n_iter=1000, verbose=True, plot=True,
                            fname_out=None, **kwargs):
    """
    Perform hierarchical bootstrap. This function performs a hierarchical bootstrap
    to test whether the means of two distributions are significantly different. This
    function is based on Saravanan et al. 2020. The functionality has been extended
    to allow for any number of hierarchical levels.

    NOTE: a p-value of 0.5 indicates that the two distributions are identical; a
    p-value close to 0 indicates that distributions_0 is greater than distributions_1;
    and a p-value close to 1 indicates that distributions_1 is greater than distributions_0.

    Parameters
    ----------
    data_0, data_1 : array
        Data arrays to be compared. Each dimension of the array represents a
        hierarchical level. For example, the first dimension could represent
        subjects, the seconds dimension could represent neurons, and the third
        dimension could represent trials.
    n_iter : int
        Number of iterations for the bootstrap.
    verbose : bool
        Whether to print the p-value.
    plot : bool
        Whether to plot the results.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    p_value : float
        p-value for the hierarchical bootstrap.
    distribution_0, distribution_1 : array
        Arrays containing the resampled means of each data array.
    bin_edges : array
        Array containing the bin edges for the joint probability matrix.
    joint_prob : array
        Array containing the joint probability matrix.
    """

    # get bootstrap samples
    distribution_0 = _get_bootstrap_distribution(data_0, n_iter=n_iter)
    distribution_1 = _get_bootstrap_distribution(data_1, n_iter=n_iter)

    # compute p-value
    p_value, joint_prob, bin_edges = _compute_p_boot(distribution_0, distribution_1)

    # print results
    if verbose:
        print(f'\np-value: {p_value:0.3f}')

    # plot results
    if plot:
        _plot_bootstrap_results(data_0, data_1, distribution_0, distribution_1,
                                bin_edges, joint_prob, fname_out, **kwargs)

    return p_value, distribution_0, distribution_1, bin_edges, joint_prob  

def _get_bootstrap_distribution(data, n_iter=1000):
    """
    Get distribution of resampled means for hierarchical bootstrap. This function
    resamples the data array and computes the mean for each iteration of the
    bootstrap.

    Parameters
    ----------
    data : array
        Data array to be resampled.
    n_iter : int
        Number of iterations.

    Returns
    -------
    distribution : array
        Array contianing the mean of each resampling iteration.
    """

    distribution = np.zeros(n_iter)
    for i_iter in range(n_iter):
        resampled_data = _resample_data(data)
        distribution[i_iter] = np.mean(resampled_data)

    return distribution
        
def _resample_data(data):
    """
    Resample data for hierarchical bootstrap. This function is used to resample
    the data array for each iteration of the bootstrap.

    Parameters
    ----------
    data : array
        Data array to be resampled.

    Returns
    -------
    resampled_data : array
        Resampled data array.
    """

    # init resampled indices
    n_dims = np.ndim(data)
    resampled_indices = np.zeros([*data.shape, n_dims], dtype=int)

    # get indices for resampling
    for i_level in range(n_dims):
        indices = _get_resampling_indices(data.shape, data.shape[i_level])
        resampled_indices[..., i_level] = indices

    # resample data 
    indices_tuple = tuple(resampled_indices[..., i] for i in range(n_dims))
    resampled_data = data[indices_tuple]

    return resampled_data


def _get_resampling_indices(shape, range):
    """
    Get resampling indices for hierarchical bootstrap. These indices are used to
    resample the data array.
    
    Parameters
    ----------
    shape : tuple
        Shape of data array.
    range : int
        Range of indices to sample from.

    Returns
    -------
    indices : array
        Array of resampled indices.
    """

    rng = np.random.default_rng()
    indices = rng.choice(range, size=shape, replace=True)
    indices = indices.astype(int)

    return indices

def _compute_p_boot(distribution_0, distribution_1, n_bins=30):    
    '''
    Compute the p-value for the hierarchical bootstrap. This function computes
    the joint probability of the two distributions and then sums the upper
    triangle of the joint probability matrix to get the p-value. A p-value of
    0.5 indicates that the two distributions are identical; a p-value close to 0
    indicates that distributions_0 is greater than distributions_1; and a p-value
    close to 1 indicates that distributions_1 is greater than distributions_0.

    This function is based on Saravanan et al. 2020 (https://github.com/soberlab/Hierarchical-Bootstrap-Paper)

    Parameters
    ----------
    distribution_0, distribution_1  : array
        Array containing the resampled means of each distribution.
    n_bins : int
        Number of bins for the joint probability matrix.

    Returns
    -------
    p_value : float
        p-value for the hierarchical bootstrap.
    joint_prob : array
        Array containing the joint probability matrix.
    bin_edges : array
        Array containing the bin edges for the joint probability matrix.
    '''

    # calculate probabilities for each distribution
    all_values = np.concatenate([distribution_0, distribution_1])
    bin_edges = np.linspace(np.min(all_values), np.max(all_values), n_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width) - (bin_width/2) # add last bin edge and shift by half bin width
    prob_0 = np.histogram(distribution_0, bins=bin_edges)[0] / len(distribution_0)
    prob_1 = np.histogram(distribution_1, bins=bin_edges)[0] / len(distribution_1)

    # compute joint probability
    joint_prob = np.outer(prob_0, prob_1)
    joint_prob = joint_prob / np.sum(joint_prob) # normalize

    # compute p-value
    p_value = np.sum(np.triu(joint_prob))

    return p_value, joint_prob, bin_edges


def _plot_bootstrap_results(data_0, data_1, distribution_0, distribution_1,
                           bin_edges, joint_prob, fname_out=None, labels=['0', '1'],
                           colors=["#d8b365", "#5ab4ac"]):
    """
    Plot bootstrap results. Plotting function for hierarchical_bootstrap().

    NOTE: the joint probability matrix is plotted using pcolormesh(),
    so the direction of the y axes is reversed, and the upper triangle of the
    matrix appears in the lower right corner of the plot.
    """

    # imports
    import matplotlib.pyplot as plt

    # create figure
    fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(18,4))

    # ax0: plot orignal distributions
    bin_edges_ = np.linspace(np.min([data_0, data_1]), np.max([data_0, data_1]), 30)
    ax0.hist(data_0.ravel(), bins=bin_edges_, color=colors[0], alpha=0.5, label=labels[0])
    ax0.hist(data_1.ravel(), bins=bin_edges_, color=colors[1], alpha=0.5, label=labels[1])
    ax0.set_xlabel('value')
    ax0.set_ylabel('count')
    ax0.set_title('Original dataset')

    # ax1: plot resampled distributions
    ax1.hist(distribution_0, bins=bin_edges, color=colors[0], alpha=0.8, label=labels[0])
    ax1.hist(distribution_1, bins=bin_edges, color=colors[1], alpha=0.8, label=labels[1])
    ax1.set_xlabel('value')
    ax1.set_ylabel('count')
    ax1.set_title('Bootstrap distributions')
    ax1.legend()

    # ax2: plot joint probability
    im = ax2.pcolormesh(bin_edges, bin_edges, joint_prob, cmap='hot')
    ax2.set_ylabel(labels[0])
    ax2.set_xlabel(labels[1])
    ax2.set_title('Joint probability')
    fig.colorbar(im, ax=ax2)

    # save figure
    if not fname_out is None:
        plt.savefig(fname_out, transparent=False)

    
    plt.show()
