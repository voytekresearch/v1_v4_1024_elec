"""Enrich odml RF

Append SNR and RF metadata to all electrodes in the odml data.

Usage:
    finalize_odml_RF.py --csv-RF=LIST \
                         --odml=FILE \
                         --out=FILE

Options:
  -h --help          Show this screen and terminate script.
  --csv-RF=FILE      RF information .csv file.
  --odml=FILE        Path to .odml template metadata file.
  --out=FILE         Output file path.

"""
from docopt import docopt
import odml
import pandas as pd
from utils import annotate_RF_elec

if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    csv_RF_path = vargs['--csv-RF']
    odml_path = vargs['--odml']
    out_path = vargs['--out']

    # Load metadata
    metadata = odml.load(odml_path)
    df = pd.read_csv(csv_RF_path)

    # Find Directions of barsweep
    stim = metadata['Recording']['Trials']['Stimulus']
    directions = stim.properties['Directions'].values

    # For all electrodes
    arrays = metadata['Arrays']
    for array in arrays.sections:
        for elec in array.sections:
            annotate_RF_elec(elec, df, directions=directions)

    # Save output
    odml.tools.XMLWriter(metadata).write_file(out_path, local_style=True)
