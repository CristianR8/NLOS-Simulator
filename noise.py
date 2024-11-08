import numpy as np

def add_sensor_noise(data, laser_intensity, noise=0.01):
    """
    Add sensor noise based on the signal-to-noise ratio (SNR).
    The noise variance is inversely proportional to the SNR.

    Args:
        data (np.ndarray): Input signal data.
        snr (float): Signal-to-noise ratio.
        laser_intensity (float): Intensity of the laser or signal power.
        
    Returns:
        np.ndarray: Noisy data.
    """
    # Calculate signal power proportional to laser intensity
    signal_power = laser_intensity  # Assume signal power ~ laser intensity

    snr = laser_intensity / noise

    # Compute noise standard deviation (std) based on SNR
    noise_std = np.sqrt(signal_power / snr)

    # Add Gaussian noise to the data
    noise = np.random.normal(0, noise_std, data.shape)
    
    # Add noise and clip to ensure non-negative values
    noisy_data = np.clip(data + noise, 0, None)
    
    return noisy_data

def add_environmental_noise(data, ambient_light=0.01):
    
    # Compute the range of uniform noise based on ambient light intensity
    noise_range = ambient_light # Scale noise with ambient light

    # Add uniform noise to the data
    noise = np.random.uniform(0, noise_range, data.shape)

    return data + noise 

def add_shot_noise(data, scale_factor=1000):
    """
    Add shot noise (Poisson noise) to the input signal data with scaling.
    
    Args:
        data (np.ndarray): Input signal data (non-negative values).
        scale_factor (float): Factor to scale the data to prevent signal loss.

    Returns:
        np.ndarray: Noisy data with added shot noise.
    """
    if np.any(data < 0):
        raise ValueError("Input data must be non-negative for shot noise.")

    # Scale the data to avoid loss of small signals
    scaled_data = data * scale_factor

    # Add Poisson noise to the scaled data
    noisy_data = np.random.poisson(scaled_data).astype(float)

    # Rescale back to the original range
    noisy_data /= scale_factor

    return noisy_data

    