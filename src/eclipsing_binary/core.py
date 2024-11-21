import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.eclipsing_binary.config import get_paths
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


class EclipsingBinary:
    def __init__(self, object_name, I_magnitude, V_magnitude, period_days,
                 epoch_of_minimum, main_eclipse_dip, second_eclipse_dip,
                 obj_type=None, RA=None, DEC=None):
        """
        Represents an eclipsing binary star system.

        Args:
            object_name (str): Name of the object.
            I_magnitude (float): I-band magnitude.
            V_magnitude (float): V-band magnitude.
            period_days (float): Orbital period in days.
            epoch_of_minimum (float): Epoch of primary minimum.
            main_eclipse_dip (float): Depth of the primary eclipse.
            second_eclipse_dip (float): Depth of the secondary eclipse.
        """
        self.object_name = object_name
        self.RA = RA
        self.DEC = DEC
        self.obj_type = obj_type
        self.I_magnitude = I_magnitude
        self.V_magnitude = V_magnitude
        self.period_days = period_days
        self.epoch_of_minimum = epoch_of_minimum
        self.main_eclipse_dip = main_eclipse_dip
        self.second_eclipse_dip = second_eclipse_dip


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

    def check_status(self):
        """
        Checks the status of the EclipsingBinary object and prints 
        information about which fields are empty and which are not.
        """

        for attr, value in self.__dict__.items():
            if attr in ('lc_I', 'lc_V'):  # Check for light curve attributes
                if value is None:
                    print(f"{attr}: Empty")
                else:
                    print(f"{attr}: Calculated")  # Just indicate presence
            elif value is None:
                print(f"{attr}: Empty")
            else:
                print(f"{attr}: {value}")
        
    def light_curves(self):
        """
        Reads and phase-folds the light curves in filters I and V.

        Args:
          lc_I_dir (str): Path to the directory containing light curves in filter I.
          lc_V_dir (str): Path to the directory containing light curves in filter V.
        """

        self.lc_I = None
        self.lc_V = None

        # Get paths from the configuration
        paths = get_paths()  # Assuming get_paths is accessible here
        lc_I_dir = os.path.join(paths['demo_data_dir'], paths['lc_I_dir'])
        lc_V_dir = os.path.join(paths['demo_data_dir'], paths['lc_V_dir'])


        try:
            # Read I-band light curve
            lc_I_file = os.path.join(lc_I_dir, f"{self.object_name}.dat")
            jd_I, mag_I, err_I = np.loadtxt(lc_I_file, unpack=True)

            # Phase-fold the I-band light curve
            phase_I = ((jd_I - self.epoch_of_minimum) / self.period_days) % 1

            # Ensure phase values are in [0, 1) even if JD < epoch_of_minimum
            phase_I = np.where(phase_I < 0, phase_I + 1, phase_I)

            self.lc_I = {
                'phase': phase_I,
                'mag': mag_I,
                'err': err_I
            }

            # Read V-band light curve only if V_magnitude is present
            if not np.isnan(self.V_magnitude):
                lc_V_file = os.path.join(lc_V_dir, f"{self.object_name}.dat")
                jd_V, mag_V, err_V = np.loadtxt(lc_V_file, unpack=True)

                # Phase-fold the V-band light curve
                phase_V = ((jd_V - self.epoch_of_minimum) / self.period_days) % 1

                # Ensure phase values are in [0, 1)
                phase_V = np.where(phase_V < 0, phase_V + 1, phase_V)

                self.lc_V = {
                    'phase': phase_V,
                    'mag': mag_V,
                    'err': err_V
                }

        except FileNotFoundError:
            print(f"Warning: Light curve file not found for {self.object_name}")

    def plot_light_curves(self, save_plot=False, plot_fit=False, n_harmonics=4):
        """
        Plots the phase-folded light curves in filters I and V.

        Args:
            save_plot (bool, optional): Whether to save the plot to a file. 
                                        Defaults to False.
        """

        if self.lc_I is None:
            print(f"Warning: No light curve data found for {self.object_name}. "
                  f"Please use the 'light_curves' method first.")  # Updated message
            return

        # Set Seaborn style
        sns.set_theme(style="ticks")

        # Create the plot with spines on all sides
        fig, ax = plt.subplots()
        ax.spines[:].set_visible(True)

        # Plot I-band light curve
        ax.errorbar(self.lc_I['phase'], self.lc_I['mag'], yerr=self.lc_I['err'], 
                    fmt='o', color='red', label='I-band', alpha=0.7)

        # Plot V-band light curve if available
        if self.lc_V is not None:
            ax.errorbar(self.lc_V['phase'], self.lc_V['mag'], yerr=self.lc_V['err'], 
                        fmt='o', color='green', label='V-band', alpha=0.7)
    

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

        if plot_fit and self.lc_I is not None and f'mag_fit_{n_harmonics}' in self.lc_I:
            ax.plot(self.lc_I[f'phase_fit_{n_harmonics}'], self.lc_I[f'mag_fit_{n_harmonics}'], '-', 
                    color='darkred', label=f'I-band Fit ({n_harmonics} harmonics)')

        if plot_fit and self.lc_V is not None and f'mag_fit_{n_harmonics}' in self.lc_V:
            ax.plot(self.lc_V[f'phase_fit_{n_harmonics}'], self.lc_V[f'mag_fit_{n_harmonics}'], '-', 
                    color='darkgreen', label=f'V-band Fit ({n_harmonics} harmonics)')


        # Save the plot if requested
        if save_plot:
            paths = get_paths()
            filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_light_curve.png")
            plt.savefig(filename)

        plt.show()

    def fit_light_curve(self, band='I', n_harmonics=4):
        """
        Fits a Fourier series model to the specified light curve.

        Args:
            band (str, optional): The band to fit ('I' or 'V'). Defaults to 'I'.
            n_harmonics (int, optional): The number of harmonics to include in the 
                                        Fourier series. Defaults to 4.
        """

        if band == 'I' and self.lc_I is not None:
            lc_data = self.lc_I
        elif band == 'V' and self.lc_V is not None:
            lc_data = self.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {self.object_name}")
            return

        # Define the model function (Fourier series)
        def fourier_model(phase, *params):
            """
            Fourier series model for the light curve.

            Args:
                phase (array-like): Phase values.
                *params: Parameters of the Fourier series (amplitude, frequency, phase).

            Returns:
                array-like: Model predictions.
            """
            result = params[0]  # Constant term
            for i in range(1, len(params) // 3):
                result += params[3*i - 2] * np.sin(2 * np.pi * i * phase + params[3*i - 1])
            return result

        # Initial guess for parameters (adjust as needed)
        initial_guess = [np.mean(lc_data['mag'])]  # Constant term
        for i in range(1, n_harmonics + 1):
            initial_guess.extend([0.1, 0, 0])  # Amplitude, frequency, phase

        # Perform the curve fit
        popt, pcov = curve_fit(fourier_model, lc_data['phase'], lc_data['mag'], p0=initial_guess)

        # Generate a smooth light curve with 100 points per period
        phase_fit = np.linspace(0, 1, 100)
        mag_fit = fourier_model(phase_fit, *popt)

        # Store the fit results in new keys within lc_data
        lc_data['phase_fit_' + str(n_harmonics)] = phase_fit  # Use a unique key
        lc_data['mag_fit_' + str(n_harmonics)] = mag_fit  # Use a unique key

    def clean_fits(self, n_harmonics_to_delete=None):
        """
        Removes fitted light curve data from the object.

        Args:
            n_harmonics_to_delete (list, optional): A list of n_harmonics values 
                                                    for which to delete the fitted data. 
                                                    Defaults to None, which deletes all fits.
        """

        if n_harmonics_to_delete is None:
            # Delete all fits
            keys_to_delete = [key for key in self.lc_I if key.startswith('phase_fit_') or key.startswith('mag_fit_')]
            if self.lc_V is not None:
                keys_to_delete += [key for key in self.lc_V if key.startswith('phase_fit_') or key.startswith('mag_fit_')]
        else:
            # Delete fits with specified n_harmonics
            keys_to_delete = []
            for n in n_harmonics_to_delete:
                keys_to_delete.extend([f'phase_fit_{n}', f'mag_fit_{n}'])

        # Remove the keys from lc_I and lc_V
        for key in keys_to_delete:
            if key in self.lc_I:
                del self.lc_I[key]
            if self.lc_V is not None and key in self.lc_V:
                del self.lc_V[key]

    def detect_outliers(self, band='I', method='all', save_plot=True):
        """
        Detects outliers in the specified light curve using the given method.

        Args:
            band (str, optional): The band to analyze ('I' or 'V'). Defaults to 'I'.
            method (str or list, optional): The outlier detection method(s) to use. 
                                           Can be 'all', 'kmeans', 'dbscan', 'isolation_forest', 
                                           or a list containing any combination of these. 
                                           Defaults to 'all'.
            save_plot (bool, optional): Whether to save the plots of clustering results.
                                        Defaults to True.
        """

        if band == 'I' and self.lc_I is not None:
            lc_data = self.lc_I
        elif band == 'V' and self.lc_V is not None:
            lc_data = self.lc_V
        else:
            print(f"Warning: No light curve data found for band {band} in {self.object_name}")
            return

        if method == 'all':
            methods = ['kmeans', 'dbscan', 'isolation_forest']
        elif isinstance(method, str):
            methods = [method]
        else:
            methods = method  # Assume method is a list

        for method in methods:
            if method == 'kmeans':
                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=10, random_state=0)  # Adjust n_clusters as needed
                lc_data['kmeans_labels'] = kmeans.fit_predict(lc_data['mag'].reshape(-1, 1))

                # Plot clustering results
                if save_plot:
                    plt.figure()
                    sns.scatterplot(x=lc_data['phase'], y=lc_data['mag'], hue=lc_data['kmeans_labels'], palette='viridis')
                    plt.title(f"{self.object_name} - {band} band - KMeans Clustering")
                    plt.xlabel("Phase")
                    plt.ylabel("Magnitude")
                    plt.gca().invert_yaxis()
                    paths = get_paths()
                    filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_{band}_kmeans.png")
                    plt.savefig(filename)
                    plt.close()

            elif method == 'dbscan':
                # Apply DBSCAN clustering
                eps = np.mean(lc_data['err'])

                dbscan = DBSCAN(eps=eps, min_samples=15)  # Adjust eps and min_samples as needed
                lc_data['dbscan_labels'] = dbscan.fit_predict(lc_data['mag'].reshape(-1, 1))

                # Plot clustering results
                if save_plot:
                    plt.figure()
                    sns.scatterplot(x=lc_data['phase'], y=lc_data['mag'], hue=lc_data['dbscan_labels'], palette='viridis')
                    plt.title(f"{self.object_name} - {band} band - DBSCAN Clustering")
                    plt.xlabel("Phase")
                    plt.ylabel("Magnitude")
                    plt.gca().invert_yaxis()
                    paths = get_paths()
                    filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_{band}_dbscan.png")
                    plt.savefig(filename)
                    plt.close()

            elif method == 'isolation_forest':
                # Apply Isolation Forest
                isolation_forest = IsolationForest(n_estimators=300)
                isolation_forest.fit(lc_data['mag'].reshape(-1, 1))
                lc_data['isolation_forest_scores'] = isolation_forest.decision_function(lc_data['mag'].reshape(-1, 1))

                # Plot anomaly scores
                if save_plot:
                    plt.figure()
                    sns.scatterplot(x=lc_data['phase'], y=lc_data['mag'], hue=lc_data['isolation_forest_scores'], palette='viridis')
                    plt.title(f"{self.object_name} - {band} band - Isolation Forest Anomaly Scores")
                    plt.xlabel("Phase")
                    plt.ylabel("Magnitude")
                    plt.gca().invert_yaxis()
                    paths = get_paths()
                    filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_{band}_isolation_forest.png")
                    plt.savefig(filename)
                    plt.close()