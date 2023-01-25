"""Enrich odml snr_rf

Append SNR and RF metadata to all electrodes in the odml data.

Usage:
    finalize_odml_RS.py --csv-RF=LIST \
                        --csv-SNR=LIST \
                        --csv-sync=LIST \
                        --odml=FILE \
                        --out=FILE

Options:
  -h --help     Show this screen and terminate script.
  --csv-RF=LIST    RF metadata csv file
  --csv-SNR=LIST   SNR metadata csv file
  --csv-sync=LIST  Synchrofact metadata csv file
  --odml=FILE      odml template file
  --out=FILE       output odml file
"""
from docopt import docopt
import odml
import pandas as pd
from utils import annotate_SNR_elec, annotate_RF_elec


if __name__ == '__main__':

    # Get arguments
    vargs = docopt(__doc__)
    SNR_path = vargs['--csv-SNR']
    RF_path = vargs['--csv-RF']
    sync_path = vargs['--csv-sync']
    odml_path = vargs['--odml']
    out_path = vargs['--out']

    # Load metadata
    metadata = odml.load(odml_path)
    df_rf = pd.read_csv(RF_path)
    df_snr = pd.read_csv(SNR_path)
    df_sync = pd.read_csv(sync_path)

    SNR_def = """Channel signal to noise ratio, measured from the response
                   to a checkerboard stimulus."""

    # For all electrodes
    arrays = metadata['Arrays']
    for array in arrays.sections:
        for elec in array.sections:
            annotate_SNR_elec(elec, df_snr)
            annotate_RF_elec(elec, df_rf)

    # Cross talk removal section
    syn_sec = odml.Section(name='Crosstalk_removal',
                           type='Crosstalk_removal',
                           definition="Removal of cross talk from the data",
                           parent=metadata)
    odml.Property(name="Iteration",
                  definition="Electrode removal iterations",
                  values=df_sync['Iteration'].values.tolist(),
                  dtype='float',
                  parent=syn_sec)
    odml.Property(name="Highest_SP",
                  definition="Highest synchrofact participation (SP) of any " +
                             "electrode for each iteration",
                  values=df_sync['Highest SP'].values.tolist(),
                  dtype='float',
                  parent=syn_sec)
    odml.Property(name="Removed_electrode_ID",
                  definition="Electrode ID removed in each iteration of the " +
                             "removal process",
                  values=df_sync['Removed electrode ID'].values.tolist(),
                  dtype='float',
                  parent=syn_sec)
    odml.Property(name="Largest_complexity",
                  definition="Largest complexity found in each iteration",
                  values=df_sync['Largest complexity'].values.tolist(),
                  dtype='float',
                  parent=syn_sec)

    # Save output
    odml.tools.XMLWriter(metadata).write_file(out_path, local_style=True)
