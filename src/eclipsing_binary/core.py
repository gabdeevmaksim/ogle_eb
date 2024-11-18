import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.eclipsing_binary.config import get_paths


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

    def plot_light_curves(self, save_plot=False):
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

        # Save the plot if requested
        if save_plot:
            paths = get_paths()
            filename = os.path.join(paths['demo_data_dir'], f"{self.object_name}_light_curve.png")
            plt.savefig(filename)

        plt.show()