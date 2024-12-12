import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from src.eclipsing_binary.config import get_paths

class EclipsingBinary:
    def __init__(self, **kwargs):  # Accept keyword arguments
        """
        Represents an eclipsing binary star system.

        Args:
            kwargs (dict): Keyword arguments for the attributes.
        """
        # Initialize attributes from keyword arguments
        for attr, value in kwargs.items():
            setattr(self, attr, value)


    @classmethod
    def from_data_row(cls, data_row):
        """
        Creates an EclipsingBinary object from a row of data.

        Args:
            data_row (numpy.void): A row from a structured NumPy array.

        Returns:
            EclipsingBinary: An instance of the EclipsingBinary class.
        """
        return cls(*data_row)  # Simplified instantiation

# src/eclipsing_binary/core.py

    def check_status(self):
        """
        Checks the status of the EclipsingBinary object and prints 
        information about which fields are empty and which are not.
        Handles arrays and dictionaries differently.
        """

        for attr, value in self.__dict__.items():
            if isinstance(value, (np.ndarray, list)):  # Check if it's an array or list
                print(f"{attr}: Array with {len(value)} elements")
            elif isinstance(value, dict):  # Check if it's a dictionary
                print(f"{attr}: Dictionary with keys: {', '.join(value.keys())}")
            elif value is None:
                print(f"{attr}: Empty")
            else:
                print(f"{attr}: {value}")
        
    def plot_light_curves(self, save_plot=False, band='both', mark_outliers=True, plot_smooth=False):
        """
        Plots the phase-folded light curves.

        Args:
            save_plot (bool, optional): Whether to save the plot to a file. 
                                        Defaults to False.
            band (str, optional): The band to plot. Can be 'I', 'V', or 'both'. 
                                  Defaults to 'both'.
            mark_outliers (bool, optional): Whether to mark outliers with a different color.
                                            Defaults to True.
            plot_smooth (bool, optional): Whether to plot the smoothed light curve.
                                          Defaults to False.
        """

        if self.lc_I is None:
            print(f"Warning: No light curve data found for {self.object_name}. "
                  f"Please use the 'light_curves' method first.")
            return

        # Set Seaborn style
        sns.set_theme(style="ticks")

        # Create the plot with spines on all sides
        fig, ax = plt.subplots()
        ax.spines[:].set_visible(True)

        if band == 'I' or band == 'both':
            # Plot I-band light curve
            if mark_outliers:
                # Plot outliers in a different color (e.g., red)
                outlier_mask = self.lc_I['outlier_mask']
                ax.errorbar(self.lc_I['phase'][~outlier_mask], self.lc_I['mag'][~outlier_mask], 
                            yerr=self.lc_I['err'][~outlier_mask], fmt='o', color='blue', label='I-band', alpha=0.7)
                ax.errorbar(self.lc_I['phase'][outlier_mask], self.lc_I['mag'][outlier_mask], 
                            yerr=self.lc_I['err'][outlier_mask], fmt='o', color='red', label='I-band Outliers', alpha=0.7)
            else:
                # Plot all data points in blue
                ax.errorbar(self.lc_I['phase'], self.lc_I['mag'], yerr=self.lc_I['err'], 
                            fmt='o', color='blue', label='I-band', alpha=0.7)
                
            # Plot smoothed I-band light curve if requested
            if plot_smooth and 'mag_smooth' in self.lc_I:
                ax.plot(self.lc_I['phase_smooth'], self.lc_I['mag_smooth'], '-', color='darkred', label='I-band Smoothed')


        if band == 'V' or band == 'both':
            # Plot V-band light curve if available
            if self.lc_V is not None:
                if mark_outliers:
                    # Plot outliers in a different color (e.g., red)
                    outlier_mask = self.lc_V['outlier_mask']
                    ax.errorbar(self.lc_V['phase'][~outlier_mask], self.lc_V['mag'][~outlier_mask], 
                                yerr=self.lc_V['err'][~outlier_mask], fmt='o', color='green', label='V-band', alpha=0.7)
                    ax.errorbar(self.lc_V['phase'][outlier_mask], self.lc_V['mag'][outlier_mask], 
                                yerr=self.lc_V['err'][outlier_mask], fmt='o', color='red', label='V-band Outliers', alpha=0.7)
                else:
                    # Plot all data points in green
                    ax.errorbar(self.lc_V['phase'], self.lc_V['mag'], yerr=self.lc_V['err'], 
                                fmt='o', color='green', label='V-band', alpha=0.7)
                    
                if plot_smooth and self.lc_V is not None and 'mag_smooth' in self.lc_V:
                    ax.plot(self.lc_V['phase_smooth'], self.lc_V['mag_smooth'], '-', color='darkgreen', label='V-band Smoothed')


        # Set labels and title
        ax.set_xlabel('Phase')
        ax.set_ylabel('Magnitude')
        
        # Invert the y-axis
        ax.invert_yaxis()

        # Calculate V-I color index
        color_index = self.V_magnitude - self.I_magnitude if not np.isnan(self.V_magnitude) else 'N/A'

        ax.set_title(f"{self.object_name}, Period: {self.period_days:.4f}, V-I: {color_index:.2f}")

        # Add legend in the upper right corner
        plt.legend(loc='upper right')

        # Save the plot if requested
        if save_plot:
            paths = get_paths()
            filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_light_curve.png")
            plt.savefig(filename)

        plt.show()

    def light_curves(self):
        """
        Reads and phase-folds the light curves in filters I and V.
        """

        self.lc_I = None
        self.lc_V = None

        # Get paths from the configuration
        paths = get_paths()
        lc_I_dir = paths['lc_I_dir']
        lc_V_dir = paths['lc_V_dir']
        try:
            # Read I-band light curve
            lc_I_file = os.path.join(lc_I_dir, f"{self.object_name}.dat")
            jd_I, mag_I, err_I = np.loadtxt(lc_I_file, unpack=True)

            # Apply outlier removal to I-band data
            self.lc_I = {
                'JD': jd_I,
                'mag': mag_I,
                'err': err_I
            }
            self.lc_I['outlier_mask'] = self.remove_outliers(band='I')  # Store outlier mask

            # Phase-fold the I-band light curve
            phase_I = ((self.lc_I['JD'] - self.epoch_of_minimum) / self.period_days) % 1
            phase_I = np.where(phase_I < 0, phase_I + 1, phase_I)
            
            # Sort I-band data by phase
            sort_indices = np.argsort(phase_I)
            self.lc_I['phase'] = phase_I[sort_indices]
            self.lc_I['mag'] = self.lc_I['mag'][sort_indices]
            self.lc_I['err'] = self.lc_I['err'][sort_indices]
            self.lc_I['JD'] = self.lc_I['JD'][sort_indices]

            # Read V-band light curve only if V_magnitude is present
            if not np.isnan(self.V_magnitude):
                lc_V_file = os.path.join(lc_V_dir, f"{self.object_name}.dat")
                jd_V, mag_V, err_V = np.loadtxt(lc_V_file, unpack=True)

                # Apply outlier removal to V-band data
                self.lc_V = {
                    'JD': jd_V,
                    'mag': mag_V,
                    'err': err_V
                }
                self.lc_V['outlier_mask'] = self.remove_outliers(band='V')  # Store outlier mask

                # Phase-fold the V-band light curve
                phase_V = ((self.lc_V['JD'] - self.epoch_of_minimum) / self.period_days) % 1
                phase_V = np.where(phase_V < 0, phase_V + 1, phase_V)

                # Sort I-band data by phase
                sort_indices = np.argsort(phase_V)
                self.lc_V['phase'] = phase_V[sort_indices]
                self.lc_V['mag'] = self.lc_V['mag'][sort_indices]
                self.lc_V['err'] = self.lc_V['err'][sort_indices]
                self.lc_V['JD'] = self.lc_V['JD'][sort_indices]

        except FileNotFoundError:
            print(f"Warning: Light curve file not found for {self.object_name}")

    def remove_outliers(self, band='I', faint_factor=10, bright_factor=2, time_threshold=0.25):
        """
        Identifies outliers in the light curve data based on the IQR method and
        neighboring points. Returns a boolean mask indicating outliers.

        Args:
            band (str, optional): The band to process ('I' or 'V'). Defaults to 'I'.
            faint_factor (float, optional): Factor to multiply IQR for faint outliers. Defaults to 10.
            bright_factor (float, optional): Factor to multiply IQR for bright outliers. Defaults to 2.
            time_threshold (float, optional): Time threshold for neighbors (in days). Defaults to 0.25.

        Returns:
            numpy.ndarray: A boolean mask where True indicates an outlier.
        """

        if band == 'I' and self.lc_I is not None:
            lc_data = self.lc_I
        elif band == 'V' and self.lc_V is not None:
            lc_data = self.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {self.object_name}")
            return None

        mag = lc_data['mag']
        jd = lc_data['JD']  # Use JD directly

        # Calculate IQR
        Q1 = np.percentile(mag, 25)
        Q3 = np.percentile(mag, 75)
        IQR = Q3 - Q1

        # Define outlier thresholds
        lower_bound = Q1 - bright_factor * IQR
        upper_bound = Q3 + faint_factor * IQR

        # Identify potential outliers
        outlier_mask = (mag < lower_bound) | (mag > upper_bound)

        # Check for neighbors in time and magnitude (revised again)
        for i in range(len(mag) - 1):
            if outlier_mask[i]:
                # Check for neighbors in time (only following measurements)
                time_neighbors = np.where((jd > jd[i]) & (np.abs(jd - jd[i]) <= time_threshold))[0]

                if len(time_neighbors) > 0:  # Only check magnitudes if time neighbors exist
                    # Compare magnitude only with time neighbors
                    mag_threshold_current = 0.3 * abs(mag[i])  # Moved mag_threshold calculation here
                    mag_neighbors = np.where(np.abs(mag[time_neighbors] - mag[i]) <= mag_threshold_current)[0]

                    # If magnitude neighbors (within time neighbors) exist, it's not an outlier
                    if len(mag_neighbors) > 0:
                        outlier_mask[i] = False

        return outlier_mask  # Return the outlier mask
    
    def smooth_light_curve(self, band='I', delta_phi=0.01, M=15, poly_degree=3):
        """
        Smooths the light curve using the Savitzky-Golay method as described in the paper.
        Handles edge cases at phase 0 and 1.

        Args:
            band (str, optional): The band to smooth ('I' or 'V'). Defaults to 'I'.
            delta_phi (float, optional): Phase window size for the initial weighted running average. 
                                        Defaults to 0.01.
            M (int, optional): Parameter for the Savitzky-Golay filter (window size = 2M+1). 
                                Defaults to 15.
            poly_degree (int, optional): Degree of the polynomial for the Savitzky-Golay filter. 
                                        Defaults to 3.
        """

        if band == 'I' and self.lc_I is not None:
          lc_data = self.lc_I
        elif band == 'V' and self.lc_V is not None:
            lc_data = self.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {self.object_name}")
            return

        phase = lc_data['phase']
        mag = lc_data['mag']
        outlier_mask = lc_data['outlier_mask']

        # Use data without outliers
        phase_filtered = phase[~outlier_mask]
        mag_filtered = mag[~outlier_mask]

        # 1. Weighted running average (handle edge cases)
        phase_averaged = np.arange(0, 1 + delta_phi, delta_phi)  # Include 1 in the range
        mag_averaged = np.zeros_like(phase_averaged)
        for i, phi in enumerate(phase_averaged):
            # Wrap around for edge cases
            window_mask = np.logical_or(
                (phase_filtered >= phi - delta_phi) & (phase_filtered <= phi + delta_phi),
                (phase_filtered >= 1 - (phi + delta_phi)) & (phase_filtered <= 1 - (phi - delta_phi))
            )
            weights = np.exp(-(phase_filtered[window_mask] - phi)**2 / (2 * delta_phi**2))
            mag_averaged[i] = np.average(mag_filtered[window_mask], weights=weights)

        # 2. Evenly sampled FLC (200 points)
        phase_even = np.linspace(0, 1, 200)
        mag_even = np.interp(phase_even, phase_averaged[:-1], mag_averaged[:-1])  # Exclude last point for interpolation

        # 3. Savitzky-Golay smoothing
        mag_smooth = savgol_filter(mag_even, window_length=2*M+1, polyorder=poly_degree)

        # Store the smoothed light curve
        lc_data['phase_smooth'] = phase_even
        lc_data['mag_smooth'] = mag_smooth

    def check_parameters(self, **kwargs):
        """
        Checks if the EclipsingBinary object's parameters meet the specified criteria.
        Handles both numeric and string parameters.

        Args:
            kwargs: Keyword arguments specifying the parameter criteria. 
                    For numeric parameters, the values should be tuples specifying 
                    the lower and upper bounds (inclusive).
                    For string parameters, the values should be the exact string 
                    or a list of possible strings.

        Returns:
            bool: True if all criteria are met, False otherwise.
        """

        for param, value_or_range in kwargs.items():
            value = getattr(self, param, None)
            if isinstance(value_or_range, tuple):  # Numeric parameter
                lower_bound, upper_bound = value_or_range
                if value is None or not (lower_bound <= value <= upper_bound):
                    return False
            else:  # String parameter
                if isinstance(value_or_range, str):
                    valid_values = [value_or_range]  # Convert single string to a list
                else:
                    valid_values = value_or_range  # Assume it's a list of valid strings
                if value not in valid_values:
                    return False
        return True

    def fit_fourier(self, band='I', non_contact=False):
        """
        Fits a Fourier series model to the specified light curve.
        Determines the most reliable number of harmonics using Baart's condition.
        Handles outlier removal using O-C residuals.

        Args:
            band (str, optional): The band to fit ('I' or 'V'). Defaults to 'I'.
            non_contact (bool, optional): Whether to force fitting even if the object 
                                          is not a contact binary. Defaults to False.
        """

        if self.obj_type != 'C' and not non_contact:
            print(f"Warning: Object {self.object_name} is not a contact binary (obj_type = {self.obj_type}). "
                  f"Skipping Fourier fitting. Use non_contact=True to force fitting.")
            return

        if band == 'I' and self.lc_I is not None:
            lc_data = self.lc_I
        elif band == 'V' and self.lc_V is not None:
            lc_data = self.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {self.object_name}")
            return
        
        # Define the model function (Fourier series)
        def fourier_model(phase, *params):
            result = params[0]  # Constant term
            for i in range(1, len(params) // 3):
                result += params[3*i - 2] * np.sin(2 * np.pi * i * phase + params[3*i - 1])
            return result

        # Function to calculate the unit-lag auto-correlation of residuals
        def calculate_autocorrelation(residuals):
            """Calculates the unit-lag auto-correlation of a sequence."""
            n = len(residuals)
            mean = np.mean(residuals)
            numerator = np.sum((residuals[:-1] - mean) * (residuals[1:] - mean))
            denominator = np.sum((residuals - mean)**2)
            return numerator / denominator

       # 1. Initial fit with a maximum number of harmonics
        phase = lc_data['phase']
        mag = lc_data['mag']
        max_harmonics = 10  # You can adjust this if needed
        initial_guess = [np.mean(mag)]
        for _ in range(max_harmonics):
            initial_guess.extend([0.1, 0, 0])
        popt, _ = curve_fit(fourier_model, phase, mag, p0=initial_guess)

        # 2. Determine the optimal number of harmonics using Baart's condition (modified)
        autocorrelations = []
        tolerance = (2 * (len(mag) - 1))**(-1/2)  # Calculate tolerance only once

        for n_harmonics in range(1, max_harmonics + 1):
            # Calculate residuals
            residuals = mag - fourier_model(phase, *popt[:3*n_harmonics])

            # Calculate auto-correlation
            autocorr = calculate_autocorrelation(residuals)
            autocorrelations.append(autocorr)

        # Find the number of harmonics with autocorrelation closest to tolerance
        optimal_harmonics = np.argmin(np.abs(np.array(autocorrelations) - tolerance)) + 1

        # 3. Remove outliers based on O-C residuals with optimal harmonics
        oc_residuals = mag - fourier_model(phase, *popt[:3*optimal_harmonics])
        Q1 = np.percentile(oc_residuals, 25)
        Q3 = np.percentile(oc_residuals, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR  # Use 1.5 IQR for both sides
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (oc_residuals < lower_bound) | (oc_residuals > upper_bound)

        # 4. Fit again without outliers using optimal harmonics
        phase_filtered = phase[~outlier_mask]
        mag_filtered = mag[~outlier_mask]
        popt_filtered, _ = curve_fit(fourier_model, phase_filtered, mag_filtered, p0=initial_guess[:3*optimal_harmonics])

        # Generate model for 100 equally spaced phases
        phase_fit = np.linspace(0, 0.99, 100)  # 100 phases from 0 to 0.99
        mag_fit = fourier_model(phase_fit, *popt_filtered)

        # Store the fit results
        lc_data[f'phase_fit_{optimal_harmonics}'] = phase_fit
        lc_data[f'mag_fit_{optimal_harmonics}'] = mag_fit
        lc_data['n_harmonics'] = optimal_harmonics

    def plot_fourier_fit(eb_object, band='I', n_harmonics=10):
        """
        Plots the phase-folded light curve and the fitted Fourier series model.

        Args:
            eb_object (EclipsingBinary): The EclipsingBinary object.
            band (str, optional): The band to plot ('I' or 'V'). Defaults to 'I'.
            n_harmonics (int, optional): The number of harmonics in the fitted model. Defaults to 10.
        """

        if band == 'I' and eb_object.lc_I is not None:
            lc_data = eb_object.lc_I
        elif band == 'V' and eb_object.lc_V is not None:
            lc_data = eb_object.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {eb_object.object_name}")
            return

        if f'mag_fit_{n_harmonics}' not in lc_data:
            print(f"Warning: Fourier fit not found for band {band} in {eb_object.object_name}. "
                f"Please use the 'fit_fourier' method first.")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(lc_data['phase'], lc_data['mag'], s=1, label='Original Data')
        plt.plot(lc_data[f"phase_fit_{n_harmonics}"], lc_data[f'mag_fit_{n_harmonics}'], color='red', label=f'Fourier Fit ({n_harmonics} harmonics)')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.title(f'{eb_object.object_name} - {band} band - Fourier Fit')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
    