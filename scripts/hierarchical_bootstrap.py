# imports
import numpy as np
import pandas as pd

# hierarchical bootstrap function

def run_hierarchical_bootstrap(df, variable, condition, level_1, level_2, n_iterations=1000,
                               verbose=True, plot=True):    
    """
    Perform hierarchical bootstrap on data. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    condition : str
        Experimental condition of interest.
    level_1 : str
        First level of hierarchy to resample.
    level_2 : str
        Second level of hierarchy to resample.
    iterations : int
        Number of iterations for resampling.
    verbose : bool
        Whether to print p-value.
    plot : bool
        Whether to plot results.

    Returns
    -------
    p_value : float
        p-value for difference between conditions.
    distribution_0 : numpy.ndarray
        Resampled distribution for condition 0.
    distribution_1 : numpy.ndarray
        Resampled distribution for condition 1.
    """

    # split groups
    df_0, df_1 = _split_experimental_conditions(df, condition)

    # run bootstrap
    distribution_0 = _hierarchical_bootstrap(df_0, variable, level_1, level_2, n_iterations)
    distribution_1 = _hierarchical_bootstrap(df_1, variable, level_1, level_2, n_iterations)

    # compute p-value
    diff = distribution_1 - distribution_0
    p_value = min(np.sum(diff > 0), np.sum(diff < 0)) / len(diff)

    # compute p-boot 
    p_boot, joint_prob, bin_edges = _compute_p_boot(distribution_0, distribution_1)

    # print/plot results    
    if verbose:
        print(f"p-value: {p_value}")
        print(f"p-boot: {p_boot}")
    if plot:
        _plot_bootstrap_results(df, variable, condition, distribution_0, distribution_1,
                               joint_prob, bin_edges)

    # return p_value, distribution_0, distribution_1
    return p_value, p_boot, joint_prob, bin_edges, distribution_0, distribution_1


def _split_experimental_conditions(df, condition):
    """
    Split dataframe into two groups based on experimental condition.
    """

    # check that there are only two experimental conditions
    conditions = df[condition].unique()
    if len(conditions) != 2:
        raise ValueError("More than two experimental conditions detected.")
        

    # split dataframe by experimental condition
    df_0 = df.loc[df[condition]==conditions[0]]
    df_1 = df.loc[df[condition]==conditions[1]]

    return df_0, df_1


def _hierarchical_bootstrap(df, variable, level_1, level_2, iterations):
    """
    Perform hierarchical bootstrap on data. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to resample.
    variable : str
        Variable to resample.
    level_1 : str
        First level of hierarchy to resample.
    level_2 : str
        Second level of hierarchy to resample.
    iterations : int
        Number of iterations for resampling.

    Returns
    -------
    distribution : numpy.ndarray
        Resampled distribution.

    """

    # get cluster info
    clusters = df[level_1].unique()
    n_clusters = len(clusters)

    # count number of instances per cluster
    instances_per_cluster = np.zeros(n_clusters)
    for i_cluster, cluster_i in enumerate(clusters):
        instances_per_cluster[i_cluster] = len(df.loc[df[level_1]==cluster_i, level_2].unique())
    n_instances = int(np.nanmean(instances_per_cluster)) # use average number of instances per cluster

    # loop through iterations
    distribution = np.zeros(iterations)
    for i_iteration in range(iterations):
        # Resample level 2 
        clusters_resampled = np.random.choice(clusters, size=n_clusters)

        # resample level 3 and get data for each cluster
        values = []
        for i_cluster, cluster_i in enumerate(clusters_resampled):
            # resample level 3
            instances = df.loc[df[level_1]==cluster_i, level_2].unique()
            instances_resampled = np.random.choice(instances, size=n_instances)

            # get data for each instance within cluster and average
            for i_instance, instance_i in enumerate(instances_resampled):
                value = df.loc[(df[level_1]==cluster_i) & (df[level_2]==instance_i), variable].values[0]
                values.append(value)

        # compute average for iteration
        distribution[i_iteration] = np.nanmean(values)

    return distribution


def _compute_p_boot(distribution_0, distribution_1, n_bins=30):    
    '''
    Compute p-value for difference between two distributions.
    This function is based on Saravanan et al. 2020.
    Source: https://github.com/soberlab/Hierarchical-Bootstrap-Paper/blob/master/Bootstrap%20Paper%20Simulation%20Figure%20Codes.ipynb
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


def _plot_bootstrap_results(df, variable, condition, distribution_0, distribution_1,
                           joint_prob, bin_edges):
    """
    Plot bootstrap results. PLotting function for run_hierarchical_bootstrap().
    """

    # imports
    import matplotlib.pyplot as plt

    # create figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))

    # ax1: plot distributions
    conditions = df[condition].unique()
    ax1.hist(distribution_0, bins=bin_edges, color='k', alpha=0.5, label=conditions[0])
    ax1.hist(distribution_1, bins=bin_edges, color='b', alpha=0.5, label=conditions[1])
    ax1.set_xlabel(variable)
    ax1.set_ylabel('count')
    ax1.set_title('Bootstrap results')
    ax1.legend()

    # ax2: plot joint probability
    im = ax2.pcolormesh(bin_edges, bin_edges, joint_prob, cmap='hot')
    ax2.set_xlabel(conditions[0])
    ax2.set_ylabel(conditions[1])
    ax2.set_title('Joint probability')
    fig.colorbar(im, ax=ax2)

    plt.show()

