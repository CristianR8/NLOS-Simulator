import numpy as np

def add_sensor_noise(data, mean=0, std=0.001):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def add_environmental_noise(data, low=0, high=0.01):
    noise = np.random.uniform(low, high, data.shape)
    return data + noise

