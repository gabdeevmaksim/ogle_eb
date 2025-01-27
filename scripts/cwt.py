import numpy as np
import fcwt
import matplotlib.pyplot as plt

def lightcurve_to_image(lightcurve, scales, wavelet_type='morlet', dj=0.1, dt=1.0):
    """
    Converts a 1D light curve to a 2D image using the Fast Continuous Wavelet Transform (FCWT).

    Args:
        lightcurve: 1D numpy array representing the light curve.
        scales: 1D numpy array of scales to use in the wavelet transform. 
               These determine the "frequencies" or "widths" analyzed.
        wavelet_type: The type of wavelet to use (e.g., 'morlet', 'paul', 'dog').
                      Default is 'morlet'.
        dj: Spacing between successive scales (if scales are not explicitly provided).
            Smaller values yield more detailed analysis but are more computationally expensive.
            Default is 0.1.
        dt: Sampling interval of the time series (light curve). Default is 1.0.

    Returns:
        A 2D numpy array representing the wavelet transform (image).
    """

    # Ensure the lightcurve is a numpy array
    lightcurve = np.asarray(lightcurve)

    # Initialize the FCWT object
    cwt_object = fcwt.FCWT(lightcurve, dt=dt, wavelet=wavelet_type, freqs=scales, p=2)

    # Perform the wavelet transform
    cwt_result = cwt_object.cwt()

    # Extract the magnitude (power) of the transform
    image = np.abs(cwt_result)

    return image

def plot_wavelet_transform(lightcurve, image, scales, dt, wavelet_name, title):
    """
    Creates a plot of the original light curve and its wavelet transform.

    Args:
        lightcurve: 1D numpy array of the light curve.
        image: 2D numpy array of the wavelet transform.
        scales: 1D numpy array of the scales used.
        dt: Sampling interval of the light curve.
        wavelet_name: Name of the wavelet function used (for the title).
        title: The title of the plot.
    """

    # Find the time axis based on the length of the light curve and sampling interval
    time = np.arange(0, len(lightcurve)) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the light curve
    ax1.plot(time, lightcurve, 'b')
    ax1.set_ylabel('Light Curve Intensity')
    ax1.set_title(title)

    # Plot the wavelet transform
    im = ax2.imshow(image, extent=[time.min(), time.max(), scales.min(), scales.max()],
                    aspect='auto', origin='lower', cmap='viridis')
    ax2.set_ylabel('Scale (Frequency)')
    ax2.set_xlabel('Time')
    ax2.set_title(f'Wavelet Transform (using {wavelet_name} wavelet)')

    # Add a colorbar for the wavelet transform
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.05)
    cbar.set_label('Magnitude')

    plt.tight_layout()
    plt.show()

# Example Usage:
# 1. Load your light curve data
# Assuming you have your light curve data in a variable called 'lightcurve_data'
# lightcurve_data = np.loadtxt("your_lightcurve_file.txt") 

# For demonstration, let's create a sample light curve:
time = np.arange(0, 100, 0.1)
lightcurve_data = np.sin(2 * np.pi * 0.5 * time) + np.sin(2 * np.pi * 2 * time)  # Example with two frequencies

# 2. Define scales
# Option 1: Define scales manually (example)
scales = np.arange(1, 64)

# Option 2: Calculate scales based on a desired resolution (dj)
# This will create log-spaced scales.
# s0 = 2 * dt  # Smallest resolvable scale
# J = int(np.log2(len(lightcurve_data) * dt / s0) / dj)  # Number of scales
# scales = s0 * 2 ** (np.arange(J + 1) * dj)

# 3. Choose wavelet type (if not using default 'morlet')
# wavelet_type = 'paul' 

# 4. Convert to image
image = lightcurve_to_image(lightcurve_data, scales)

# 5. Visualize (optional)
plot_wavelet_transform(lightcurve_data, image, scales, dt=0.1, wavelet_name='morlet', title='Light Curve and Wavelet Transform')

# 6. Save the image (optional)
# np.save("lightcurve_image.npy", image)