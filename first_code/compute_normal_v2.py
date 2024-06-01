import numpy as np

def nmz(x):
    """Normalize vector"""
    return x / np.linalg.norm(x)

def compute_normal_v2(objects):
    """
    Compute the normal vector and update the objects list.
    
    Parameters:
    objects (list): List of objects where each object is a list containing vectors.
    
    Returns:
    list: Updated objects list with the normal vector computed.
    """
    normal_vector = -nmz(np.cross(objects[2], objects[3]))
    
    if np.linalg.norm(objects[1] + normal_vector) > np.linalg.norm(objects[1]):
        normal_vector = -normal_vector
    
    objects[4] = normal_vector
    return objects

