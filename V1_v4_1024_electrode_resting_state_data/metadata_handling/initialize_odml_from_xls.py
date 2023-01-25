"""XLS to ODML

Creates a basic metadata odml template from some pre-filled .xls spreadsheets.

Usage:
    calculate_SNR.py --equipment=FILE \
                     --subject=FILE \
                     --out=FILE


Options:
    -h --help         Show this screen and terminate script.
    --equipment=FILE  Path to .xls file with equipment metadata.
    --subject=FILE    Path to .xls file with subject specific metadata.
    --out=FILE        Output file path.
"""

from docopt import docopt
import odmltables.odml_xls_table as odxlstable

if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    equipment_path = vargs['--equipment']
    subject_path = vargs['--subject']
    output_odml = vargs['--out']

    # create OdmlXlsTable object
    subject = odxlstable.OdmlXlsTable()
    setup = odxlstable.OdmlXlsTable()

    # loading the data
    subject.load_from_xls_table(subject_path)
    setup.load_from_xls_table(equipment_path)

    # merge into a single odml
    subject.merge(setup, overwrite_values=False)

    # generate output file
    subject.write2odml(output_odml)
