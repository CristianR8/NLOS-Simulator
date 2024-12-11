# Function to save measurements to .mat format
import scipy.io
from io import BytesIO

def save_to_mat(measurements, params):
    mat_data = {
        'measurements': measurements,
        'params': params
    }
    buffer = BytesIO()
    scipy.io.savemat(buffer, mat_data)
    buffer.seek(0)
    return buffer

# Function to save measurements to .raw format
def save_to_raw(measurements):
    buffer = BytesIO()
    buffer.write(measurements.tobytes())
    buffer.seek(0)
    return buffer