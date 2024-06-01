import numpy as np
from compute_normal_v2 import compute_normal_v2
def create_object(x, y, z, width, height, theta, kind, kind2):
    """
    Create an object with given parameters.
    
    Parameters:
    x, y, z (float): Coordinates of the object.
    width (float): Width of the object.
    height (float): Height of the object.
    theta (float): Angle for the object's orientation.
    kind (str): Type of the object.
    kind2 (str): Additional type information.
    
    Returns:
    list: Object with computed properties.
    """
    object = [None] * 5
    
    if kind2 == 'back':
        if kind == 'wall':
            object[0] = 'wall'
            object[1] = np.array([x, y, z])
            
            facetX = width * np.cos(theta) - 0 * np.sin(theta)
            facetY = width * np.sin(theta) + 0 * np.cos(theta)
            
            if abs(facetX) < 1e-6: facetX = 0
            if abs(facetY) < 1e-6: facetY = 0
            
            object[2] = np.array([facetX, facetY, 0])
            object[3] = np.array([0, 0, height])
            object = compute_normal_v2(object)
    else:
        if kind == 'wall':
            object[0] = 'wall'
            
            facetX = width * np.cos(theta) - 0 * np.sin(theta)
            facetY = width * np.sin(theta) + 0 * np.cos(theta)
            
            if abs(facetX) < 1e-6: facetX = 0
            if abs(facetY) < 1e-6: facetY = 0
            
            object[2] = np.array([facetX, facetY, 0])
            object[1] = np.array([x - facetX / 2, y - facetY / 2, z])
            object[3] = np.array([0, 0, height])
            object = compute_normal_v2(object)
    
    return object


