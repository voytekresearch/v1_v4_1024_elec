"""Threshold crossings

Extract threshold crossings from the raw analog signals

Usage:
    highpass_ns6.py --ns6=FILE --out=FILE

Options:
    -h --help     Show this screen and terminate script.
    --ns6=FILE    Path to .ns6 data file.
    --out=FILE    Output .npy file path.
"""
from docopt import docopt
from utils import anasig_from_nsp
from elephant.signal_processing import butter
import gc
import quantities as pq
from numpy.lib.format import open_memmap

if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    path_ns6 = vargs['--ns6']
    path_npy = vargs['--out']

    hipassed = None
    # 1. Get the analogsignal of each index (i.e. electrode)
    for i, anasig in enumerate(anasig_from_nsp(path_ns6)):

        # Create a memmap (if it does not exist yet)
        if hipassed is None:
            hipassed = open_memmap(path_npy, dtype='float32', mode='w+',
                                   shape=(128, anasig.shape[0]))

        # 2. Filter the signal between 250Hz and 9000Hz
        anasig = butter(anasig,
                        highpass_freq=250.0*pq.Hz,
                        lowpass_freq=9000.0*pq.Hz,
                        fs=anasig.sampling_rate.magnitude)
        gc.collect()

        # 3. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig,
                            highpass_freq=(fq + 2)*pq.Hz,
                            lowpass_freq=(fq - 2)*pq.Hz,
                            fs=anasig.sampling_rate.magnitude)
            gc.collect()

        # Assign analogsignal to the memmap and release memory
        sig = anasig.magnitude.flatten()
        hipassed[i, :] = sig.astype('float32')
        del anasig, sig
        gc.collect()

        # Flush memmap changes to disk before continueing
        hipassed.flush()
