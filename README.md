# Eclipsing Binary Analysis

This repository contains code and data for analyzing eclipsing binary stars from the OGLE survey.

## Project Goals

  * Read and process eclipsing binary data from fixed-width format files.
  * Cross-match OGLE stars with identification data to extract object type, RA, and DEC.
  * Read and process extinction data.
  * Phase-fold light curves using orbital period data.
  * Visualize the phase-folded light curves.
  * Fit light curves with Fourier series models.
  * Perform cross-matching with Gaia DR3 data.

## Data

  * Eclipsing binary data from the OGLE survey (`data/OGLE_SMC_ECL.txt`).
  * Identification data (`data/smc_ident.dat`).
  * Extinction data (`data/smc_ext.dat`).
  * Light curve data in I and V filters (`data/light_curves/lc_I` and `data/light_curves/lc_V`).

## Code

  * `src/eclipsing_binary/core.py`:
      * `EclipsingBinary` class: Represents an eclipsing binary star system.
          * `light_curves()`: Reads and phase-folds light curves.
          * `plot_light_curves()`: Plots the light curves.
          * `check_status()`: Prints information about the object's attributes.
          * `fit_light_curve()`: Fits a Fourier series model to the light curve.
          * `smooth_light_curve()`: Smooths the light curve using the Savitzky-Golay method.
  * `src/eclipsing_binary/utils.py`:
      * `read_data()`: Reads and combines data from multiple files.
      * `gaia_cross_match()`: Cross-matches with Gaia DR3 data.
  * `src/eclipsing_binary/config.py`:
      * `get_paths()`: Reads data paths from the `config.ini` file.

## Usage

1.  Clone the repository: `git clone https://github.com/your-username/your-repo-name.git`
2.  Install the required packages: `pip install astropy astroquery matplotlib numpy seaborn scikit-learn`
3.  Configure the data paths in `config.ini`.
4.  Run the `analysis.py` script to perform the analysis and generate plots.

## Example

```python
from src.eclipsing_binary.utils import read_data
from src.eclipsing_binary.config import get_paths

# Get data paths from config.ini
paths = get_paths()

# Read the data
binaries = read_data(paths)

# Process and plot the first eclipsing binary
binaries[0].light_curves()
binaries[0].plot_light_curves(save_plot=True)
binaries[0].check_status()
```

## Contributing

Contributions are welcome\! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/url?sa=E&source=gmail&q=https://www.google.com/url?sa=E%26source=gmail%26q=https://www.google.com/url?sa=E%26source=gmail%26q=LICENSE) file for details.