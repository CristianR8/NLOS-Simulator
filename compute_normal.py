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
    objects[k][4] = -nmz(np.cross(objects[k][2], objects[k][3])) 
    
    if np.linalg.norm(objects[k][1] + objects[k][4]) > np.linalg.norm(objects[k][1]):
        objects[k][4] = -objects[k][4]
    
    return objects


