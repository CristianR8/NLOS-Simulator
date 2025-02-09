import numpy as np

def add_sensor_noise(data, SNR_dB):
    
    # Calculate signal power
    signal_power = np.mean(np.abs(data) ** 2)
    
    # Calculate noise power based on desired SNR
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    
    # Generate Gaussian noise
    noise_std = np.sqrt(noise_power)
    noise = abs(noise_std * np.random.normal(size = data.shape))
    
    #Add noise to the original signal
    noisy_data = data + noise
    
    return noisy_data

def add_background_noise(data, sbr, axis=-1, inplace=False):
    """
    Add background noise to transient data to achieve a desired Signal-to-Background Ratio (SBR).

    Parameters:
    - data: np.ndarray
        Transient data (e.g., photon counts) with time bins along the specified axis.
    - sbr: float
        Desired signal-to-background ratio (SBR). Must be > 0.
    - axis: int
        Axis corresponding to the time bins. Default is -1.
    - inplace: bool
        If True, modify the data in place. Otherwise, return a new array with background added.

    Returns:
    - np.ndarray
        Transient data with background noise added.
    """

    if sbr <= 0:
        raise ValueError("SBR must be greater than 0.")

    # Sum of photons along the specified axis (signal)
    signal_sum = np.sum(data, axis=axis, keepdims=True)

    # Calculate the number of ambient photons needed to achieve the desired SBR
    background_photon_sum = signal_sum / sbr

    # Uniform background to add to each time bin
    background = background_photon_sum / data.shape[axis]

    if inplace:
        data += background
        return data
    else:
        return data + background

def add_poisson_noise(data, n_mc_samples=1, scale_factor=100):
    """
    Add Poisson noise to the transient data with scaling to preserve the signal.

    Parameters:
    - transient: np.ndarray
        The input transient data (non-negative values).
    - n_mc_samples: int
        The number of Monte Carlo samples to generate.
    - scale_factor: float
        Factor to scale the transient data to avoid excessive noise.

    Returns:
    - np.ndarray
        Transient data with added Poisson noise, shape (n_mc_samples, *transient.shape).
    """

    #if np.any(data < 0):
        #raise ValueError("Input data must be non-negative for Poisson noise.")

    # Scale the data to avoid loss of small signals (Poisson noise works best with larger values)
    scaled_data = data * scale_factor

    # Add Poisson noise to the scaled data
    noisy_data = np.random.poisson(lam=scaled_data, size=(n_mc_samples,) + data.shape).astype(float)

    # Rescale the noisy data back to the original range
    noisy_data /= scale_factor

    if n_mc_samples == 1:
        return noisy_data.squeeze(axis=0)
    return noisy_data

