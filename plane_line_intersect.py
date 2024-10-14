import numpy as np

def plane_line_intersect(n, V0, P0, P1):
    """
    Computes the intersection of a plane and a segment (or a straight line)
    
    Parameters:
    n (numpy.ndarray): Normal vector of the Plane
    V0 (numpy.ndarray): Any point that belongs to the Plane
    P0 (numpy.ndarray): End point 1 of the segment P0P1
    P1 (numpy.ndarray): End point 2 of the segment P0P1
    
    Returns:
    tuple: (I, check)
        I (numpy.ndarray): The point of intersection
        check (int): Indicator
            0 => disjoint (no intersection)
            1 => the plane intersects P0P1 in the unique point I
            2 => the segment lies in the plane
            3 => the intersection lies outside the segment P0P1
    """
    I = np.zeros(3)
    u = P1 - P0
    w = P0 - V0
    D = np.dot(n, u)
    N = -np.dot(n, w)
    check = 0

    if abs(D) < 1e-7:  # The segment is parallel to plane
        if N == 0:     # The segment lies in plane
            check = 2
            return I, check
        else:
            check = 0  # no intersection
            return I, check

    # Compute the intersection parameter
    sI = N / D
    I = P0 + np.dot(sI, u) 
    if sI < 0 or sI > 1:
        check = 3  # The intersection point lies outside the segment, so there is no intersection
    else:
        check = 1

    return I, check

