import numpy as np
from src.eclipsing_binary.core import EclipsingBinary

def read_data(filename, ident_filename=None):
    """
    Reads a file with fixed-width columns containing eclipsing binary data,
    optionally combines it with identification data, and returns a list of 
    EclipsingBinary objects.

    Args:
      filename (str): The path to the main data file.
      ident_filename (str, optional): The path to the identification data file.

    Returns:
      list: A list of EclipsingBinary objects.
    """

    column_names = ['object_name', 'I_magnitude', 'V_magnitude', 'period_days', 
                    'epoch_of_minimum', 'main_eclipse_dip', 'second_eclipse_dip']
    
    # Define the widths of each column
    col_widths = [19, 7, 7, 13, 12, 6, 6]  

    eclipsing_binaries = []

    with open(filename, 'r') as file:
        for line in file:
            # Split the line into fields based on column widths
            values = [line[sum(col_widths[:i]):sum(col_widths[:i+1])].strip() 
                      for i in range(len(col_widths))]

            # Convert numeric values to floats, handling '-'
            for i in range(1, len(values)):  # Start from index 1 to skip 'object_name'
                try:
                    values[i] = float(values[i])
                except ValueError:
                    if values[i] == '-':
                        values[i] = np.nan  # Replace '-' with np.nan
                    else:
                        raise  # Raise the error for other conversion issues

            # Create an EclipsingBinary object and add it to the list
            eclipsing_binaries.append(EclipsingBinary(*values))  # Pass values directly
    
        if ident_filename:
          ident_data = {}
          with open(ident_filename, 'r') as ident_file:
              for line in ident_file:
                  parts = line.split()
                  ident_data[parts[0]] = {
                      'obj_type': parts[1],
                      'RA': parts[2],
                      'DEC': parts[3]
                  }

          # Add identification data to EclipsingBinary objects
          for binary in eclipsing_binaries:
              if binary.object_name in ident_data:
                  binary.obj_type = ident_data[binary.object_name]['obj_type']
                  binary.RA = ident_data[binary.object_name]['RA']
                  binary.DEC = ident_data[binary.object_name]['DEC']

    return eclipsing_binaries

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from src.eclipsing_binary.config import get_paths
import os

def test_dbscan(eb_object, eps_coef_range, min_samples_range, band='I', save_plot=True):
    """
    Tests DBSCAN with different parameters on an EclipsingBinary object.

    Args:
        eb_object (EclipsingBinary): The EclipsingBinary object to analyze.
        eps_coef_range (list): A list of three values [start, end, step] for the 
                                 coefficient to multiply with the mean error to get eps.
        min_samples_range (list): A list of three values [start, end, step] for the 
                                    min_samples parameter of DBSCAN.
        band (str, optional): The band to analyze ('I' or 'V'). Defaults to 'I'.
        save_plot (bool, optional): Whether to save the plots. Defaults to True.
    """

    if band == 'I' and eb_object.lc_I is not None:
        lc_data = eb_object.lc_I
    elif band == 'V' and eb_object.lc_V is not None:
        lc_data = eb_object.lc_V
    else:
        print(f"Warning: No light curve data found for band {band} in {eb_object.object_name}")
        return

    # Calculate base eps (mean of errors)
    base_eps = np.mean(lc_data['err'])

    # Iterate through eps and min_samples values
    for eps_coef in np.arange(*eps_coef_range):
        for min_samples in np.arange(*min_samples_range):
            # Calculate eps
            eps = base_eps * eps_coef

            # Apply DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(lc_data['mag'].reshape(-1, 1))

            # Plot the results
            if save_plot:
                plt.figure()
                sns.scatterplot(x=lc_data['phase'], y=lc_data['mag'], hue=labels, palette='viridis')
                plt.title(f"{eb_object.object_name} - {band} band - DBSCAN (eps_coef={eps_coef:.2f}, min_samples={min_samples})")
                plt.xlabel("Phase")
                plt.ylabel("Magnitude")
                plt.gca().invert_yaxis()
                paths = get_paths()  # Assuming get_paths is accessible here
                filename = os.path.join(paths['demo_data_dir'], f"{eb_object.object_name}_{band}_dbscan_{eps_coef:.2f}_{min_samples}.png")
                plt.savefig(filename)
                plt.close()

def sigma_clip_window(eb_object, window_range, sigma_range, band='I', save_plot=True):
    """
    Applies sigma-clipping with a window function to an EclipsingBinary object.

    Args:
        eb_object (EclipsingBinary): The EclipsingBinary object to analyze.
        window_range (list): A list of three values [start, end, step] for the 
                               window size in phase units.
        sigma_range (list): A list of three values [start, end, step] for the 
                             number of sigmas to use as a threshold.
        band (str, optional): The band to analyze ('I' or 'V'). Defaults to 'I'.
        save_plot (bool, optional): Whether to save the plots. Defaults to True.
    """

    if band == 'I' and eb_object.lc_I is not None:
        lc_data = eb_object.lc_I
    elif band == 'V' and eb_object.lc_V is not None:
        lc_data = eb_object.lc_V
    else:
        print(f"Warning: No light curve data found for band {band} in {eb_object.object_name}")
        return

    phase = lc_data['phase']
    mag = lc_data['mag']
    err = lc_data['err']

    for window_size in np.arange(*window_range):
        for sigma in np.arange(*sigma_range):
            # Apply window function and sigma-clipping
            phase_window = np.arange(0, 1, window_size)
            mag_filtered = np.array([])
            phase_filtered = np.array([])
            err_filtered = np.array([])
            outlier_indices = np.array([], dtype=bool)  # To store outlier indices

            for i in range(len(phase_window) - 1):
                # Define the window boundaries
                lower_bound = phase_window[i]
                upper_bound = phase_window[i + 1]

                # Select data within the window
                mask = (phase >= lower_bound) & (phase < upper_bound)
                window_data = mag[mask]
                window_err = err[mask]

                # Apply sigma-clipping within the window
                median = np.median(window_data)
                std = np.median(window_data)
                filtered_mask = np.abs(window_data - median) <= sigma * std
                filtered_window_data = window_data[filtered_mask]

                # Store filtered data and corresponding phases/errors
                mag_filtered = np.append(mag_filtered, filtered_window_data)
                phase_filtered = np.append(phase_filtered, phase[mask][filtered_mask])
                err_filtered = np.append(err_filtered, window_err[filtered_mask])
                outlier_indices = np.append(outlier_indices, ~filtered_mask)  # Store outlier indices

            # Ensure the last part of the light curve is included
            last_window_mask = phase >= phase_window[-1]
            mag_filtered = np.append(mag_filtered, mag[last_window_mask])
            phase_filtered = np.append(phase_filtered, phase[last_window_mask])
            err_filtered = np.append(err_filtered, err[last_window_mask])
            outlier_indices = np.append(outlier_indices, np.zeros(sum(last_window_mask), dtype=bool))

            # Plot the results
            if save_plot:
                plt.figure()

                # Plot all data points in blue
                sns.scatterplot(x=phase, y=mag, color='blue', label='Data')

                # Overlay outliers in red
                sns.scatterplot(x=phase[outlier_indices], y=mag[outlier_indices], color='red', label='Outliers')

                for i in range(len(phase_window) - 1):
                    lower_bound = phase_window[i]
                    upper_bound = phase_window[i + 1]
                    mask = (phase >= lower_bound) & (phase < upper_bound)

                    # Calculate and plot median and std lines
                    median = np.median(mag[mask])
                    std = np.median(err[mask])
                    plt.hlines(y=median, xmin=lower_bound, xmax=upper_bound, color='green', linewidth=1)
                    plt.hlines(y=median + sigma * std, xmin=lower_bound, xmax=upper_bound, color='orange', linestyle='--', linewidth=0.5)
                    plt.hlines(y=median - sigma * std, xmin=lower_bound, xmax=upper_bound, color='orange', linestyle='--', linewidth=0.5)


                for boundary in phase_window:
                    plt.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.5)

                plt.title(f"{eb_object.object_name} - {band} band - Sigma Clipping (window={window_size:.2f}, sigma={sigma})")
                plt.xlabel("Phase")
                plt.ylabel("Magnitude")
                plt.gca().invert_yaxis()
                plt.legend()
                paths = get_paths()
                filename = os.path.join(paths['demo_data_dir'], f"{eb_object.object_name}_{band}_sigma_clip_{window_size:.2f}_{sigma}.png")
                plt.savefig(filename)
                plt.close()