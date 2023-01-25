"""Calculate MUAe

Calculate the Multiunit Activity envelope (MUAe) per channel from the raw
analog signal.

Usage:
    calculate_MUA.py --ns6=FILE --out=FILE --odml=FILE --array=INT
    calculate_MUA.py --ns6=FILE --out=FILE --odml=FILE --array=INT --eyesig=FILE

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
from datetime import datetime
import gc
import os
import warnings
import scipy


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    path_ns6 = vargs['--ns6']
    array_id = vargs['--array']
    odmlpath = vargs['--odml']
    path_muae = vargs['--out']
    try:
        path_eye = vargs['--eyesig']
    except KeyError:
        path_eye = None
        warnings.warn('No eyesignals found.')

    muae = []
    # 1. Get the analogsignal of each index (i.e. electrode)
    for anasig in anasig_from_array(path_ns6, array_id, odmlpath=odmlpath):

        # 2. Filter the signal between 500Hz and 9000Hz
        anasig = butter(anasig,
                        highpass_freq=500.0*pq.Hz,
                        lowpass_freq=9000.0*pq.Hz,
                        fs=anasig.sampling_rate.magnitude)
        gc.collect()

        # 3. Rectify the filtered wave
        anasig = anasig.rectify()
        gc.collect()

        # 4. Low pass filter at 200Hz
        anasig = butter(anasig,
                        lowpass_freq=200.0*pq.Hz,
                        fs=anasig.sampling_rate.magnitude)
        gc.collect()

        # 5. Downsample signal from 30kHz to 1kHz resolution (factor 30)
        anasig = anasig.downsample(30, ftype='fir')
        gc.collect()

        # 6. Bandstop filter the 50, 100 and 150 Hz frequencies
        # Compensates for artifacts from the European electric grid
        for fq in [50, 100, 150]:
            anasig = butter(anasig,
                            highpass_freq=(fq + 2)*pq.Hz,
                            lowpass_freq=(fq - 2)*pq.Hz,
                            fs=anasig.sampling_rate.magnitude)
            gc.collect()

        muae.append(anasig)

    # Use custom function to merge analogsignals
    muae = merge_anasiglist(muae)

    # Create empty neo blocks for new datafiles
    date = datetime.today().strftime('%Y-%m-%d')
    array_block = neo.Block(name='MUAe signal from ' + path_ns6,
                            date_of_creation=date)
    array_block.segments = [neo.Segment()]

    # Enrich with analog signal data
    array_block.segments[0].analogsignals = []
    array_block.segments[0].analogsignals.append(muae)

    # Mark epochs, based on metadata
    array_block = mark_epochs(array_block, odmlpath, eyepath=path_eye)

    # Delete file if existing
    if os.path.isfile(path_muae):
        os.remove(path_muae)

    # Save to nix
    print(array_block)
    print(array_block.annotations)
    outFile = NixIO(path_muae, mode='ow')
    outFile.write(array_block)
    outFile.close()

    # Save to mat
    matdict = {'data': muae.magnitude, **muae.array_annotations}
    scipy.io.savemat(path_muae.replace('nix', 'mat'), matdict)
