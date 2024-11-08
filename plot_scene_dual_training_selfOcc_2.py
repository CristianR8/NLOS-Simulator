import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plane_line_intersect import plane_line_intersect
import trimesh

def plot_scene_dual_training_selfOcc_2(laser_pos, objects, params):
    # Crear una escena y agregar todas las mallas
    scene = trimesh.Scene(objects)
    
    # Agregar la posición del láser
    laser_sphere = trimesh.creation.icosphere(radius=0.05)
    laser_sphere.apply_translation(laser_pos)
    scene.add_geometry(laser_sphere)
    
    # Agregar los píxeles de la cámara
    cam_pixel_dim = params['cam_pixel_dim']
    pixel_x = np.linspace(
        params['camera_FOV_center'][0] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * cam_pixel_dim),
        params['camera_FOV_center'][0] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * cam_pixel_dim),
        cam_pixel_dim
    )
    pixel_y = np.linspace(
        params['camera_FOV_center'][1] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * cam_pixel_dim),
        params['camera_FOV_center'][1] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * cam_pixel_dim),
        cam_pixel_dim
    )
    X, Y = np.meshgrid(pixel_x, pixel_y)
    cam_pos = np.vstack([X.ravel(), Y.ravel(), np.zeros(cam_pixel_dim**2)]).T

    camera_pixels = trimesh.points.PointCloud(cam_pos, colors=[0, 255, 0])
    scene.add_geometry(camera_pixels)
    
    # Visualizar la escena
    scene.show()