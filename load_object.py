import trimesh
import numpy as np

def load_3d_object(filename, x, y, z, scale=1.0, theta=0, center_mesh=True):
    """
    Carga un objeto 3D desde un archivo .obj, aplica transformaciones y prepara el objeto para la simulación.
    """
    # Carga la malla usando trimesh
    mesh = trimesh.load(filename, process=True)
    
    # Opcional: Centrar la malla en su origen antes de aplicar transformaciones
    if center_mesh:
        mesh.apply_translation(-mesh.centroid)
    
    # Aplicar escalado uniforme
    mesh.apply_scale(scale)
    
    # Crear una matriz de rotación para el ángulo especificado (alrededor del eje Z)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        theta, [0, 0, 1], point=[0, 0, 0]
    )
    mesh.apply_transform(rotation_matrix)
    
    # Trasladar la malla a la posición deseada (x, y, z)
    mesh.apply_translation([x, y, z])
    
    # Asegurarse de que las normales de las caras están disponibles
    if not mesh.face_normals.any():
        mesh.recompute_face_normals()
    
    # Crear el intersector de rayos con el motor especificado
    if ray_engine == 'pyembree':
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    
    # Preparamos el diccionario con la información necesaria
    mesh_data = {
        'type': 'mesh',
        'mesh': mesh,
        'intersector': intersector
    }
    
    return mesh_data