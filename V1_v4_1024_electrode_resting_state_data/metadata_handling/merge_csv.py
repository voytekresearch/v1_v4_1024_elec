"""Merge CSV

Merges a list of given csv files into a single csv file.

Usage:
    merge_csv.py --csv-list=LIST --out=FILE

Options:
  -h --help        Show this screen and terminate script.
  --csv-list=LIST  List of csv files.
  --out=FILE       Output file path.

"""
from docopt import docopt

import pandas as pd

if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    csv_path_list = [arg for arg in vargs['--csv-list'].split(' ')]
    out_path = vargs['--out']

    # Create a list of dataframes
    lst_df = []
    for csv in csv_path_list:
        df = pd.read_csv(csv)
        lst_df.append(df)

    # Concatenate
    result = pd.concat(lst_df)

    # Save result
    result.to_csv(out_path, index=False)
