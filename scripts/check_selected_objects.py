import os
import pandas as pd
from eclipsing_binary.config import get_paths

def check_light_curves_exist(ident_file, lc_dir):
    """
    Checks how many light curves from the ident_file exist in the lc_dir folder.

    Args:
        ident_file (str): The path to the identification file containing the object names.
        lc_dir (str): The directory where the light curve files are stored.

    Returns:
        int: The number of existing light curve files.
    """
    # Read the identification data
    ident_col_specs = [(0, 19), (20, 24), (25, 36), (37, 49), (50, 66), (67, 82), (83, 99), (100, 116)]
    ident_names = ['object_name', 'type', 'RA_coord', 'DEC_coord', 'OGLE-IV', 'OGLE-III', 'OGLE-II', 'other_names']
    ident_df = pd.read_fwf(ident_file, colspecs=ident_col_specs, names=ident_names, header=None)

    # Check for the existence of each LC file
    existing_lcs = 0
    for object_name in ident_df['object_name']:
        lc_file = os.path.join(lc_dir, f"{object_name}.dat")
        if os.path.isfile(lc_file):
            existing_lcs += 1

    return existing_lcs

if __name__ == "__main__":
    # Load the necessary paths
    paths = get_paths()
    ident_file = paths['ident_file']
    lc_I_dir = paths['lc_I_dir']

    # Check how many LCs exist
    num_existing_lcs = check_light_curves_exist(ident_file, lc_I_dir)
    print(f"Number of existing light curves: {num_existing_lcs}")