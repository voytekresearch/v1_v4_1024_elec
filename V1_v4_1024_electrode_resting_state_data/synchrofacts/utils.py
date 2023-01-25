import os
import pickle
import numpy as np
import quantities as pq
import warnings
import gc
from elephant.spike_train_synchrony import Synchrotool
from elephant.spike_train_surrogates import surrogates
from joblib import Parallel, delayed
import random


def get_syncounts(sts, path=False, bins=None):

    warnings.simplefilter("ignore", category=UserWarning)

    # Calculate complexity time histogram
    synobj = Synchrotool(sts, sampling_rate=sts[0].sampling_rate, spread=1)
    synobj.time_histogram = None  # Remove the heaviest part
    if path:
        with open(path, 'wb') as f:
            pickle.dump(synobj, f)

    # Count
    synobj.epoch.array_annotations['complexity'] = \
        synobj.epoch.array_annotations['complexity'].astype(np.uint16)
    cpx = synobj.epoch.array_annotations['complexity']
    if bins is None:
        bins = np.arange(0.5, np.max(cpx)+1.5, step=1)
    tot_syncounts, _ = np.histogram(cpx, bins=bins)

    # Get the syncount histogram electrode-wise
    synobj.annotate_synchrofacts()
    elec_syncounts = []
    for st in synobj.input_spiketrains:
        cpx_ind = st.array_annotations['complexity']
        ind_syncounts, _ = np.histogram(cpx_ind, bins=bins)
        elec_syncounts.append(ind_syncounts)
    elec_syncounts = np.stack(elec_syncounts, axis=-1)

    del synobj, cpx, cpx_ind
    gc.collect()

    return tot_syncounts, elec_syncounts, bins


def calc_surr(sts, bins=None, seed=None, dt=5*pq.ms):
    # Restart the random seed, important to avoid same results in threads
    np.random.seed(seed)
    random.seed(np.random.randint(2**32))

    # Generate surrogates
    surrs = []
    for st in sts:
        surr = surrogates(spiketrain=st,
                          surr_method='dither_spike_train',
                          dt=dt)[0]

        # Discretize the spikes, very important for synchrofact removal!
        tu = surr.times.units
        mag_sampling_rate = surr.sampling_rate.rescale(1/tu).magnitude.flatten()
        mag_times = surr.times.magnitude.flatten()
        discrete_times = (mag_times // (1 / mag_sampling_rate)
                          / mag_sampling_rate)
        discrete_times *= tu
        discrete_surr = surr.duplicate_with_new_data(discrete_times)
        discrete_surr.annotations = surr.annotations
        surrs.append(discrete_surr)

    # Calculate the complexities
    tot_syncounts, elec_syncounts, _ = get_syncounts(surrs, bins=bins)

    # Cleanup
    del surrs, bins
    gc.collect()

    return (tot_syncounts, elec_syncounts)


def get_surrogate_syncounts(sts, bins, n_surr=1000):
    # Generate and process surrogates
    # Function yields up to 1/n_surr resolution of probability
    # dt is the surrogate dither parameter

    # Generate random seeds
    seeds = np.random.randint(2**32-1, size=n_surr)

    # Execute parallel surrogate calculation
    with Parallel(n_jobs=int(os.cpu_count()), prefer='processes',
                  backend='multiprocessing') as parallel:
        tupl_lst = parallel(delayed(calc_surr)(sts, bins, seed)
                            for seed in seeds)

    # Unpack list of tuples
    surr_tot_syncount_lst = [tupl[0] for tupl in tupl_lst]
    surr_el_syncount_lst = [tupl[1] for tupl in tupl_lst]

    # Convert surrogate syncounts to arrays
    surr_tot_syncount_arr = np.array(surr_tot_syncount_lst)
    del surr_tot_syncount_lst
    gc.collect()

    surr_el_syncount_arr = np.stack(surr_el_syncount_lst, axis=-1)
    del surr_el_syncount_lst
    gc.collect()

    return surr_tot_syncount_arr, surr_el_syncount_arr
