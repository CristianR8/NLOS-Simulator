import trimesh
import numpy as np

def load_3d_object(filename, scale=1.0, position=np.array([0, 0, 0])):
    """
    Load a 3D object from an .obj file and apply transformations.
    """
    mesh = trimesh.load(filename)
    mesh.apply_scale(scale)
    mesh.apply_translation(position)
    return mesh