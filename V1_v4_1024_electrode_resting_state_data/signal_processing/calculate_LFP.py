"""Calculate LFP

Calculate the Local Field Potential (LFP) per channel from the raw
analog signal.

Usage:
    calculate_LFP.py --ns6=FILE --out=FILE --odml=FILE --array=INT
    calculate_LFP.py --ns6=FILE --out=FILE --odml=FILE --array=INT --eyesig=FILE

Options:
    -h --help     Show this screen and terminate script.
    --ns6=FILE    Path to .ns6 data file.
    --odml=FILE   Path to .odml metadata file.
    --array=INT   Number of the analized array.
    --eyesig=FILE Path to the eye signals
    --out=FILE    Output file path.
"""
from docopt import docopt
from utils import anasig_from_array, merge_anasiglist
from utils import mark_epochs
import quantities as pq
import neo
from neo.io.nixio import NixIO
from elephant.signal_processing import butter
import gc
import warnings
import scipy

if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    path_ns6 = vargs['--ns6']
    array_id = vargs['--array']
    odmlpath = vargs['--odml']
    path_lfp = vargs['--out']
    try:
        path_eye = vargs['--eyesig']
    except KeyError:
        path_eye = None
        warnings.warn('No eyesignals found.')

    lfp = []
    # 1. Get the analogsignal of each index (i.e. electrode)
    for anasig in anasig_from_array(path_ns6, array_id, odmlpath=odmlpath):

        # 2. Filter the signal between 1Hz and 150Hz
        anasig = butter(anasig, lowpass_freq=150.0*pq.Hz)
        gc.collect()

        # 3. Downsample signal from 30kHz to 500Hz resolution (factor 60)
        anasig = anasig.downsample(60, ftype='fir')
        gc.collect()

        # 4. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig,
                            highpass_freq=(fq + 2)*pq.Hz,
                            lowpass_freq=(fq - 2)*pq.Hz)
            gc.collect()

        lfp.append(anasig)

    # Use custom function to merge analogsignals
    lfp = merge_anasiglist(lfp)

    # Create empty neo blocks for new datafiles
    array_block = neo.Block(name='LFP signal from ' + path_ns6)
    array_block.segments = [neo.Segment()]

    # Enrich with analog signal data
    array_block.segments[0].analogsignals = []
    array_block.segments[0].analogsignals.append(lfp)

    # Mark epochs, based on metadata
    array_block = mark_epochs(array_block, odmlpath, eyepath=path_eye)

    # Save to nix
    outFile = NixIO(path_lfp, mode='ow')
    outFile.write(array_block)
    outFile.close()

    # Save to mat
    matdict = {'lfp': lfp.magnitude, **lfp.array_annotations}
    scipy.io.savemat(path_lfp.replace('nix', 'mat'), matdict)
