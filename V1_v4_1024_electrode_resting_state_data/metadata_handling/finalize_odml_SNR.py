"""Enrich odml snr_rf

Append SNR and RF metadata to all electrodes in the odml data.

Usage:
    enrich_odml_snr.py --csv-SNR=LIST \
                       --odml=FILE \
                       --out=FILE

Options:
  -h --help          Show this screen and terminate script.
  --csv-SNR=LIST     SNR information .csv file.
  --odml=FILE        Path to .odml template metadata file.
  --out=FILE         Output file path.

"""
from docopt import docopt
import odml
import pandas as pd
from utils import annotate_SNR_elec

if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    csv_SNR_path = vargs['--csv-SNR']
    odml_path = vargs['--odml']
    out_path = vargs['--out']

    # Load metadata
    metadata = odml.load(odml_path)
    df = pd.read_csv(csv_SNR_path)

    SNR_def = """Channel signal to noise ratio, measured from the response
                   to a checkerboard stimulus."""

    # For all electrodes
    arrays = metadata['Arrays']
    for array in arrays.sections:
        for elec in array.sections:
            annotate_SNR_elec(elec, df)

    # Save output
    odml.tools.XMLWriter(metadata).write_file(out_path, local_style=True)
