import os
import sys

import pandas as pd
import numpy as np
from eclipsing_binary.utils import read_data
from eclipsing_binary.config import get_paths
from eclipsing_binary.core import EclipsingBinary

def create_folded_light_curves(output_file):
    """
    Reads selected objects from the input file, creates light curves folded with their orbital periods,
    and writes the results to the output file with the specified header.

    Args:
        input_file (str): The path to the input file containing the selected objects.
        output_file (str): The path to the output file where the folded light curves will be written.

    Returns:
        None
    """
    # Load the necessary paths
    os.getcwd()
    paths = get_paths()
    
    # Read the data
    eclipsing_binaries = read_data(paths)

    # Create folded light curves and write to file
    with open(output_file, 'w') as f:
        f.write("Period,NumPoints,Class\n")
        for eb in eclipsing_binaries:
            eb.light_curves()
            if eb.lc_I is not None:
                period = eb.period_days
                num_points = len(eb.lc_I['phase'])
                lc_class = eb.obj_type
                f.write(f"{period},{num_points},{lc_class}\n")
                for phase, mag in zip(eb.lc_I['phase'], eb.lc_I['mag']):
                    f.write(f"{phase},{mag}\n")

if __name__ == "__main__":
    output_file = 'folded_light_curves.txt'
    create_folded_light_curves(output_file)
    print(f"Folded light curves created and written to {output_file}")