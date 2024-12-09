import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        lc_I_dir = paths['lc_I_dir']
        lc_V_dir = paths['lc_V_dir']
        print(lc_I_dir, lc_V_dir)

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

    def plot_light_curves(self, save_plot=False, plot_fit=False, n_harmonics=4, band = 'both'):
        """
        Plots the phase-folded light curves and optionally the fitted model.

        Args:
            save_plot (bool, optional): Whether to save the plot to a file. 
                                        Defaults to False.
            plot_fit (bool, optional): Whether to plot the fitted model. 
                                       Defaults to False.
            n_harmonics (int, optional): The number of harmonics used in the 
                                         fitted model to plot. Defaults to 4.
            band (str, optional): The band to plot. Can be 'I', 'V', or 'both'. 
                                  Defaults to 'both'.
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

        if band == 'I' or band == 'both':
            # Plot I-band light curve
            ax.errorbar(self.lc_I['phase'], self.lc_I['mag'], yerr=self.lc_I['err'], 
                        fmt='o', color='red', label='I-band', alpha=0.7)

            # Plot the fitted model for I-band (if requested)
            if plot_fit and f'mag_fit_{n_harmonics}' in self.lc_I:
                ax.plot(self.lc_I[f'phase_fit_{n_harmonics}'], self.lc_I[f'mag_fit_{n_harmonics}'], '-', 
                        color='darkred', label=f'I-band Fit ({n_harmonics} harmonics)')

        if band == 'V' or band == 'both':
            # Plot V-band light curve if available
            if self.lc_V is not None:
                ax.errorbar(self.lc_V['phase'], self.lc_V['mag'], yerr=self.lc_V['err'], 
                            fmt='o', color='green', label='V-band', alpha=0.7)

                # Plot the fitted model for V-band (if requested)
                if plot_fit and f'mag_fit_{n_harmonics}' in self.lc_V:
                    ax.plot(self.lc_V[f'phase_fit_{n_harmonics}'], self.lc_V[f'mag_fit_{n_harmonics}'], '-', 
                            color='darkgreen', label=f'V-band Fit ({n_harmonics} harmonics)')

    

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
