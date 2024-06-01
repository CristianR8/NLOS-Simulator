import numpy as np

def nmz(x):
    """Normalize vector"""
    return x / np.linalg.norm(x)

def compute_normal(objects, k):
    """
    Compute the normal vector and update the objects list.
    
    Parameters:
    objects (list): List of objects where each object is a list containing vectors.
    k (int): Index of the object in the objects list.
    
    Returns:
    list: Updated objects list with the normal vector computed for the k-th object.
    """
    normal_vector = -nmz(np.cross(objects[k][2], objects[k][3]))
    
    if np.linalg.norm(objects[k][1] + normal_vector) > np.linalg.norm(objects[k][1]):
        normal_vector = -normal_vector
    
    objects[k][4] = normal_vector
    return objects


