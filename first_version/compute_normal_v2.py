import numpy as np

def nmz(x):
    """ Normalize a vector """
    return x / np.linalg.norm(x)

def compute_normal_v2(objects):
    """
    Compute the normal vector for the given object based on cross-product of the direction vectors.
    
    Parameters:
    - objects: A list representing an object with its properties (position, direction, etc.)
    
    Returns:
    - objects: The same list with the computed normal vector stored in the fifth element (objects[4])
    """
    # Calculate the normal vector
    objects[4] = -nmz(np.cross(objects[2], objects[3]))  # objects[4] corresponds to objects{5} in MATLAB

    # Check the norm condition and possibly flip the normal vector
    if np.linalg.norm(objects[1] + objects[4]) > np.linalg.norm(objects[1]):
        objects[4] = -objects[4]

    return objects
