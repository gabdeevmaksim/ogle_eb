import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('./src/')) 

from eclipsing_binary.config import get_paths
from eclipsing_binary.utils import read_data
from eclipsing_binary.core import EclipsingBinary

def main():
    """
    Reads data, fits Fourier series to contact binaries, and saves the results.
    """

    # 1. Read paths from config.ini
    paths = get_paths()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(paths['demo_data_dir'], 'I_fit')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Read data to the class objects
    binaries = read_data(paths)

    # 3. Sort out contact binaries
    contact_binaries = [binary for binary in binaries if binary.check_parameters(obj_type='C')]

    # 4. Fit and plot for each contact binary
    for binary in contact_binaries:
        try:
            # Load and fit light curve
            binary.light_curves()
            binary.fit_fourier(band='I')

            # Plot the light curve and fitted model using plot_fourier_fit
            plot_filename = os.path.join(output_dir, f"{binary.object_name}_fourier_fit.png")
            binary.plot_fourier_fit(band='I', filename=plot_filename)

            # Save the fitted curve to a CSV file
            n_harmonics = binary.lc_I['n_harmonics']
            csv_filename = os.path.join(output_dir, f"{binary.object_name}_fourier_fit.csv")
            df = pd.DataFrame({
                'phase': binary.lc_I[f'phase_fit_{n_harmonics}'],
                'mag': binary.lc_I[f'mag_fit_{n_harmonics}']
            })
            df.to_csv(csv_filename, index=False)

        except Exception as e:
            print(f"Error processing {binary.object_name}: {e}")

if __name__ == "__main__":
    main()