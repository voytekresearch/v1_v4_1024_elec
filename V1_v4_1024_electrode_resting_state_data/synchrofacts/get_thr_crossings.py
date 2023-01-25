"""covariance matrix calculation
Original authors of methods: Alexander Kleinjohann, Simon Essink
Script author: Aitor Morales-Gregorio

Usage:
  get_thr_crossings_mpi.py <infile> <eigenvalues> <eigenvectors> <clean_npy> <ns6> <out>

"""
from docopt import docopt
import numpy as np
import neo
import quantities as pq
import pickle


def extract_spikes(data, sampling_rate, threshold_multiplier):
    # Quiroga 2004 threshold
    threshold = -threshold_multiplier * np.median(np.abs(data) / 0.6745)
    data_binarized = (data < threshold).astype(int)
    idx_threshold_crossings = np.nonzero(np.diff(data_binarized) > 0)[0]
    spike_times = []
    spike_waveforms = []
    for i in idx_threshold_crossings:
        spike_times.append(i/sampling_rate)
        spike_waveforms.append(data[i-20:i+25])
    return np.array(spike_times), np.array(spike_waveforms)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    input_file = arguments['<infile>']
    output_vals = arguments['<eigenvalues>']
    output_vecs = arguments['<eigenvectors>']
    clean_data = arguments['<clean_npy>']
    path_ns6 = arguments['<ns6>']
    path_out = arguments['<out>']

    mmap = np.load(input_file, mmap_mode='r', allow_pickle=True)
    n_channels, n_samples = mmap.shape

    # actual computation, data has already been centered
    corrmat = np.zeros((n_channels, n_channels))
    triu_indices = np.triu_indices(n_channels)
    for ii, jj in zip(*triu_indices):
        corrmat[ii, jj] = np.dot(mmap[ii], mmap[jj])

    # normalise the covariances
    channel_sum = np.sum(mmap, axis=1)
    channel_sum_squared = np.sum(mmap**2, axis=1)
    corrmat /= n_samples
    corrmat[triu_indices[1], triu_indices[0]] = corrmat[triu_indices]
    print('Correlation matrix calculated.')

    # Eigendecomposition
    eigen_vals, eigen_vecs = np.linalg.eig(corrmat)
    order = np.argsort(eigen_vals)[-1::-1]
    eigen_vals = eigen_vals[order]
    eigen_vecs = eigen_vecs[:, order]
    np.save(output_vals, eigen_vals)
    np.save(output_vecs, eigen_vecs)
    print('Eigendecomposition of covariance matrix done.')

    # Get metadata from ns6 file without loading a the signals
    reader = neo.io.BlackrockIO(filename=path_ns6)
    block = reader.read_block(lazy=True)
    anasig_proxy = block.segments[0].analogsignals[-1]
    sampling_rate = anasig_proxy.sampling_rate.rescale('Hz').magnitude
    len_data, ch_per_nsp = anasig_proxy.shape
    t_stop = anasig_proxy.t_stop
    print('Metadata obtained.')

    # Remove first PC from data
    mmap = np.dot(mmap.T, eigen_vecs).astype('float32')
    eigen_vecs[:, 0] = 0  # Remove first PC
    mmap = np.dot(mmap, eigen_vecs.T).T.astype('float32')
    np.save(clean_data, mmap)
    print('First principal component removed.')

    # Extract threshold crossings
    spiketrains = []
    for i_ch in range(ch_per_nsp):
        sts, sws = extract_spikes(mmap[i_ch, :],
                                  sampling_rate,
                                  threshold_multiplier=5)
        name = anasig_proxy.array_annotations['channel_names'][i_ch]
        st = neo.SpikeTrain(np.array(sts)*pq.s, t_stop=t_stop,
                            waveforms=np.array(sws),
                            sampling_rate=anasig_proxy.sampling_rate,
                            elec_id=name.split('-')[-1],
                            array_id=name.split('-')[0][4:])
        spiketrains.append(st)
    print('All threshold crossings extracted.')

    with open(path_out, 'wb') as f:
        pickle.dump(spiketrains, f)
