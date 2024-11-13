import numpy as np
from compute_normal_v2 import compute_normal_v2

def create_object(x, y, z, width, height, theta, kind, kind2):
    """
    Create an object based on the parameters provided, mimicking the MATLAB create_object function.
    
    Parameters:
    - x, y, z: Coordinates of the object
    - width: Width of the object
    - height: Height of the object
    - theta: Angle of rotation
    - kind: Type of object (e.g., 'wall')
    - kind2: Secondary type (e.g., 'back')

    Returns:
    - object: A list representing the object with its properties
    """
    # Initialize the object with 5 empty elements
    object = [None] * 5
     
    if kind2 == 'back':
        if kind == 'wall':
            object[0] = 'wall'
            object[1] = np.array([x, y, z])
            
            facetX = width * np.cos(theta) - 0 * np.sin(theta)
            facetY = width * np.sin(theta) + 0 * np.cos(theta)
            
            # Handle floating-point precision issues
            if abs(facetX) < 1e-6:
                facetX = 0
            if abs(facetY) < 1e-6:
                facetY = 0
            
            object[2] = np.array([facetX, facetY, 0])  # Direction of the wall
            object[3] = np.array([0, 0, height])  # Height direction
            object = compute_normal_v2(object)
    
    else:
        if kind == 'wall':
            object[0] = 'wall'
            
            facetX = width * np.cos(theta) - 0 * np.sin(theta)
            facetY = width * np.sin(theta) + 0 * np.cos(theta)
            
            # Handle floating-point precision issues
            if abs(facetX) < 1e-6:
                facetX = 0
            if abs(facetY) < 1e-6:
                facetY = 0
            
            object[2] = np.array([facetX, facetY, 0])  # Direction of the wall
            object[1] = np.array([x - facetX / 2, y - facetY / 2, z])  # Center the object correctly
            object[3] = np.array([0, 0, height])  # Height direction
            object = compute_normal_v2(object)

    return object
