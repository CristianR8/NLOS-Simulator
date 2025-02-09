import trimesh
import numpy as np

# Load the OBJ file
mesh = trimesh.load('objects/duck.obj')

# Define the rotation matrix for 90 degrees around the Y-axis
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.radians(-90),  # Convert degrees to radians using numpy
    direction=[1, 0, 0],  # Y-axis
    point=mesh.centroid  # Rotate around the center of the mesh
)

# Apply the rotation to the mesh
mesh.apply_transform(rotation_matrix)

# Save the rotated mesh to a new OBJ file
mesh.export('duck.obj')

print("Rotation completed and saved to 'dog.obj'.")