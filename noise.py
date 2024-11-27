import numpy as np

def add_sensor_noise(data, SNR_dB):
    signal_power = np.mean(data ** 2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise_std = np.sqrt(noise_power)
    noise = noise_std * np.random.randn(*data.shape)
    noise_clipped = np.clip(noise, 0, None)
    y_meas_vec_noisy = data + noise_clipped
    return y_meas_vec_noisy

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

    