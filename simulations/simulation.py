import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
#from scipy.io import savemat
import time
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
from trimesh.visual.material import SimpleMaterial
from noise import add_sensor_noise, add_environmental_noise, add_shot_noise
from PIL import Image
import io
import uuid
import glob
import os

c = 299792458

object_folder = '/home/cristianr/NLOS-Simulator/objects/'

# Function to list all .obj files in the folder
def get_obj_files(folder):
    return [os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.obj'))]


def simulation(camera_FOV, cam_pixel_dim, bin_size, laser_intensity, w, xcoord, ycoord, obj_file): 

    objects = []
    
    # SPAD Camera Params
    camera_FOV = camera_FOV
    camera_FOV_center = [0, -camera_FOV / 2, 0]
    cam_pixel_dim = cam_pixel_dim
    bin_size = bin_size
    FOV_radius = camera_FOV / cam_pixel_dim  
 

    # Laser params
    laser_intensity = laser_intensity
    laser_pos = np.array([0, 0, 0])
    laser_normal = np.array([0, 0, 1])

    # Simulation parameters
    wall_discr = c / 2 * bin_size / 4

    # Room Dimensions (in meters)
    xmin, xmax = -3/2, 3/2
    ymin, ymax = 0, 3
    zmin, zmax = 0, 3

    params = {
        'cam_pixel_dim': cam_pixel_dim,
        'camera_FOV': camera_FOV,
        'camera_FOV_center': camera_FOV_center,
        'FOV_radius': FOV_radius,
        'laser_intensity': laser_intensity,
        'bin_size': bin_size,
        'c': c,
        'laser_pos': laser_pos,
        'laser_normal': laser_normal,
        'wall_discr': wall_discr,
        }
    
    # Define colors for the scene elements
    right_color = [86, 101, 115, 255] 
    left_color = [128, 139, 150, 255]  
    back_color = [40, 55, 71, 255]  
    ceiling_color = [52, 73, 94, 255]  
    object_color = [241, 148, 138, 255]  
    laser_color = [255, 0, 0, 255]  
    camera_pixel_color = [0, 255, 0, 255]
    front_wall_color = [200, 200, 200, 30]  

    furthest_scene_point = np.array([xmax, ymax, zmax])
    furthest_spad_point = np.array([-params['camera_FOV'] / 2, -params['camera_FOV'], 0])
    d1 = np.linalg.norm(furthest_scene_point - laser_pos)
    d2 = np.linalg.norm(furthest_spad_point - furthest_scene_point)
    max_dist_travel = d1 + d2
    num_time_bins = int(np.ceil((max_dist_travel / c / bin_size + 0.2 * max_dist_travel / c / bin_size)))
    params['num_time_bins'] = num_time_bins

    # Crear una pared como malla
    def create_wall_mesh(origin, u_vec, v_vec, color ):
        # Calcular el ancho y alto de la pared
        width = np.linalg.norm(u_vec)
        height = np.linalg.norm(v_vec)
        
        # Crear una caja delgada como pared
        wall = trimesh.creation.box(extents=(width, 0.01, height))  # Grosor de 0.01 en el eje Y
        wall.visual.vertex_colors = color

        
        # Calcular la rotación necesaria para alinear la pared
        normal = np.cross(u_vec, v_vec)
        normal = normal / np.linalg.norm(normal)
        y_axis = np.array([0, 1, 0])  # Eje Y, ya que la pared se creó en XZ plane
        rotation_axis = np.cross(y_axis, normal)
        rotation_angle = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            wall.apply_transform(rotation_matrix)
        
        # Posicionar la pared en la escena
        wall.apply_translation(origin + 0.5 * (u_vec + v_vec))
        return wall

    # Crear paredes y techo como mallas
    back_wall = create_wall_mesh(np.array([xmin, ymax, 0]), np.array([xmax - xmin, 0, 0]), np.array([0, 0, zmax]), back_color)
    right_wall = create_wall_mesh(np.array([xmax, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax]), right_color)
    left_wall = create_wall_mesh(np.array([xmin, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax]), left_color)
    ceiling = create_wall_mesh(np.array([xmin, ymax, zmax]), np.array([xmax - xmin, 0, 0]), np.array([0, -ymax, 0]), ceiling_color)


    # Lista de objetos
    objects.extend([back_wall, right_wall, left_wall, ceiling])

    # Agregar la faceta plana
    w = w  # Ancho
    h = 1.1  # Altura
    xcoord_list = [xcoord]
    ycoord_list = [ycoord]

    for xx in range(len(xcoord_list)):
        v1 = np.array([xcoord_list[xx], ycoord_list[xx], 0])
        u = np.array([1, 0, 0])
        #v_dir = v1 - np.array([0, 0, 0])
        #cos_theta = np.dot(u, v_dir) / (np.linalg.norm(u) * np.linalg.norm(v_dir))
        #cos_theta = np.clip(cos_theta, -1, 1)
        theta = -np.clip(np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1)), -1, 1)

        #theta_rad = -np.arccos(cos_theta)
        obj_path = os.path.join(object_folder, obj_file)
        obj = trimesh.load(obj_path, force='mesh')
        obj_extents = obj.extents
        scale_factors = [w / obj_extents[0], h / obj_extents[2]]
        scale_factor = min(scale_factors) * 2
        obj.apply_scale(scale_factor)
        rotation = rotation_matrix(np.radians(90), [1, 0, 0])
        obj.apply_transform(rotation)
        rotation_z = rotation_matrix(theta, [0, 0, 1])
        z_min = obj.vertices[:, 2].min()
        obj.apply_transform(rotation_z)
        obj.apply_translation([0,0, -z_min])
        obj.apply_translation(v1)
        obj.visual.vertex_colors = object_color 
        objects.append(obj)
        
        
    # Crear una escena y agregar todas las mallas
    scene = trimesh.Scene()
    scene.add_geometry(objects)
    front_rotation = rotation_matrix(np.radians(90), [1, 0, 0])
    second_rotation = rotation_matrix(np.radians(180), [0, 0, 1])
    scene.apply_transform(front_rotation)
    scene.apply_transform(second_rotation) 
        
    # Agregar la posición del láser
    laser_sphere = trimesh.creation.icosphere(radius=0.04)
    laser_sphere.apply_translation(laser_pos)
    laser_sphere.visual.vertex_colors = laser_color
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

    pixel_t = np.linspace(0, (params['num_time_bins'] - 1) * params['bin_size'], params['num_time_bins'])

    X, Y = np.meshgrid(pixel_x, pixel_y)
    cam_pos = np.vstack([X.ravel(), Y.ravel(), np.zeros(cam_pixel_dim**2)]).T

    X_ind, Y_ind = np.meshgrid(range(params['cam_pixel_dim']), range(params['cam_pixel_dim']), indexing='xy')
    cam_pos_ind = np.vstack([X_ind.ravel(), Y_ind.ravel()]).T

    X, Y, T = np.meshgrid(pixel_x, pixel_y, pixel_t)
    cam_pixel = np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

    sphere_radius = 0.02  
    camera_pixel = []
    rotation = rotation_matrix(np.radians(90), [1, 0, 0])
    for pos in cam_pos:
        sphere = trimesh.creation.icosphere(radius=sphere_radius)
        sphere.apply_translation(pos)
        sphere.apply_transform(rotation)
        sphere.front_wall_color = [200, 200, 200, 100]  
        sphere.visual.vertex_colors = camera_pixel_color
        camera_pixel.append(sphere)
        
    # Agregar todas las esferas a la escena
    scene.add_geometry(camera_pixel)
    
    ########################################################## Front wall
    # Define the wall dimensions and position
    x_start = 0          # Start at x = 0
    x_end = xmax         # End at x = xmax
    y_pos = ymin         # The wall is at the front (minimum y)
    z_start = zmin       # Start at z = zmin
    z_end = zmax         # End at z = zmax

    # Define grid resolution (adjust as needed for density)
    x_steps = 20         # Number of points along x-axis
    z_steps = 20         # Number of points along z-axis

    # Create a grid of points
    x_coords = np.linspace(x_start, x_end, x_steps)
    z_coords = np.linspace(z_start, z_end, z_steps)
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords)
    Y_grid = np.full_like(X_grid, y_pos)  # All y coordinates are at y_pos

    # Flatten the grids to create a list of point coordinates
    wall_points = np.vstack((X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten())).T

    # Create small spheres at each point
    sphere_radius = 0.02  # Adjust the radius as needed
    wall_spheres = []
    rotation_wall = rotation_matrix(np.radians(-90), [1, 0, 0])
    
    for pos in wall_points:
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
        sphere.apply_translation(pos)
        sphere.apply_transform(rotation_wall)
        sphere.visual.vertex_colors = [200, 200, 200, 255]  # Color of the spheres
        wall_spheres.append(sphere)

    # Add the spheres representing the wall to the scene
    scene.add_geometry(wall_spheres)
    
    # Visualizar la escena
    scene_filename = f"/home/cristianr/NLOS-Simulator/3D_scene/scene_{uuid.uuid4()}.obj"
    scene.export(scene_filename)
    
    # Combinar las mallas y crear el intersector de rayos
    combined_mesh = trimesh.util.concatenate(objects)
    try:
        ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(combined_mesh)
    except ImportError:
        ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(combined_mesh)

    # Obtener triángulos y normales
    triangles = combined_mesh.triangles
    triangle_normals = combined_mesh.face_normals
    num_triangles = len(triangles)

    # Inicializar el vector de mediciones
    num_pixels = params['cam_pixel_dim'] ** 2
    num_bins = params['num_time_bins']

    # Crear el vector de medición
    y_meas_vec = np.zeros(cam_pixel.shape[0])

    fourpi = 4 * np.pi * np.pi
    floor_normal = np.array([0, 0, 1])
    floor_pixel_width = params['camera_FOV'] / params['cam_pixel_dim']

    # Iterar sobre cada triángulo de la malla (equivalente a scene_pixel en el código original)
    for idx in range(len(triangles)):
        # Obtener el triángulo y su normal
        triangle = triangles[idx]
        normal = triangle_normals[idx]
        area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        scene_center = triangle.mean(axis=0)

        m = (scene_center[1] - cam_pos[:, 1]) / (scene_center[0] - cam_pos[:, 0]) 
        b = scene_center[1] - np.dot(m, scene_center[0])
        
        xint = -b / m
        
        noc = xint > 0
        if np.sum(noc) > 0:
            lps = params['laser_pos'] - scene_center
            fovsp = cam_pos[noc, :] - scene_center

            d1s = np.sum(lps**2)
            d2s = np.sum(fovsp**2, axis=1)
            d1 = np.sqrt(d1s)
            d2 = np.sqrt(d2s)
            distance = d1 + d2
            tbin = distance / (c * params['bin_size'])
            arrival_bin = np.ceil(tbin).astype(int)

            dot1 = np.maximum(0, np.sum(np.dot(normal, lps / d1))) 
            dot2 = np.maximum(0, np.sum(normal * fovsp / d2[:, np.newaxis], axis=1))            
            dot3 = np.maximum(0, np.sum(np.dot(params['laser_normal'], -lps / d1)))  
            dot4 = np.maximum(0, np.sum(floor_normal * -fovsp / d2[:, np.newaxis], axis=1))

            intensity = params['laser_intensity'] * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * d2s)

            coord = np.dot(arrival_bin - 1, params['cam_pixel_dim']**2) + np.dot((cam_pos_ind[noc, 0] - 1), params['cam_pixel_dim']) + cam_pos_ind[noc, 1]
            
            y_meas_vec[coord] += intensity
            
    # Reshape para visualización
    y_meas_vec = y_meas_vec.reshape((params['cam_pixel_dim'], params['cam_pixel_dim'], num_bins), order='F')


    # Añadir ruido de sensor
    y_meas_vec_noisy = add_sensor_noise(y_meas_vec, laser_intensity=laser_intensity)

    # Añadir ruido ambiental
    y_meas_vec_noisy_2 = add_environmental_noise(y_meas_vec_noisy)

    # Añadir ruido de disparo (shot noise)
    y_meas_vec_noisy_3 = add_shot_noise(y_meas_vec_noisy_2) 

    # Visualizar la respuesta temporal de un píxel de muestra
    plt.figure()
    plt.plot(y_meas_vec[0, 0, :])
    plt.title('Temporal response in a specific pixel')
    plt.xlabel('time bin')
    plt.ylabel('intensity')
    temporal_response_plot = plt.gcf()

    # Sumar sobre los bins de tiempo para obtener una imagen 2D
    y_sum = np.sum(y_meas_vec, axis=2)
    y_sum = np.roll(y_sum, shift=1, axis=-1)
    plt.figure()
    plt.imshow(y_sum, cmap='viridis')
    plt.colorbar()
    plt.title('Intensidad Integrada en el Tiempo')
    plt.gca().invert_yaxis()
    plt.xlabel('Píxel X')
    plt.ylabel('Píxel Y')
    plt.show()

    # Guardar los datos
    #filename = f"Simulacion_Con_Malla_{int(params['bin_size'] * 1e12)}ps.mat"
    #savemat(filename, {'params': params, 'y_meas_vec': y_meas_vec_noisy_reshaped, 'objects': objects})
    return scene_filename, temporal_response_plot

obj_files = get_obj_files(object_folder)

# Define Gradio inputs and outputs
inputs = [
    gr.Slider(0.1, 1.0, value=0.25, label="Camera FOV"),
    gr.Slider(16, 64, step=1, value=32, label="Camera Pixel Dimension"),
    gr.Number(value=390e-12, label="Bin Size"),
    gr.Slider(50, 1000, value=1000, label="Laser Intensity (mW)"),
    gr.Slider(0.1, 0.4, value=0.2, label="Object Size"),    
    gr.Number(value=-0.5, label="Object X Coordinates (comma-separated)"),
    gr.Number(value=1.25, label="Object Y Coordinates (comma-separated)"),
    gr.Dropdown(obj_files, label="Select 3D Object"),
    #gr.Slider(5, 200, step=1, value=20, label="SNR"),

]

outputs = [
    gr.Model3D(label="Scene Visualization"),
    gr.Plot(label="Temporal Response Plot"),
]

# Launch Gradio Interface
interface = gr.Interface(
    fn=simulation,
    inputs=inputs,
    outputs=outputs,
    description="Interactive simulation for SPAD camera and laser scene"
)

interface.launch()