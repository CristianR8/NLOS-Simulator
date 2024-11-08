import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
from trimesh.visual.material import SimpleMaterial
from noise import add_sensor_noise, add_environmental_noise, add_shot_noise
import uuid
import glob
import os
import base64
import logging
import plotly.graph_objects as go
from check_overlaps import check_overlaps
from streamlit_plotly_events import plotly_events


c = 299792458

object_folder = '/home/cristianr/NLOS-Simulator/objects/'

# Function to list all .obj files in the folder
def get_obj_files(folder):
    return [os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.obj'))]

def simulation(camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls): 
    objects = []
    scene_objects = []
    
    # SPAD Camera Params
    camera_FOV_center = [0, -camera_FOV / 2, 0]
    FOV_radius = camera_FOV / cam_pixel_dim  
    
    # Laser params
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
    
    # Function to create wall mesh
    def create_wall_mesh(origin, u_vec, v_vec, color ):
        width = np.linalg.norm(u_vec)
        height = np.linalg.norm(v_vec)
        
        wall = trimesh.creation.box(extents=(width, 0.01, height))  # Thickness of 0.01 in Y-axis
        wall.visual.vertex_colors = color if color else [255, 255, 255, 255]
        
        # Calculate rotation to align wall
        normal = np.cross(u_vec, v_vec)
        normal = normal / np.linalg.norm(normal)
        y_axis = np.array([0, 1, 0])  # Y-axis
        rotation_axis = np.cross(y_axis, normal)
        rotation_angle = np.arccos(np.clip(np.dot(y_axis, normal), -1.0, 1.0))
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_matrix_wall = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            wall.apply_transform(rotation_matrix_wall)
        
        # Position the wall in the scene
        wall.apply_translation(origin + 0.5 * (u_vec + v_vec))
        return wall
    
    if not hide_walls: 
        # Create walls and ceiling
        back_wall = create_wall_mesh(np.array([xmin, ymax, 0]),
                                    np.array([xmax - xmin, 0, 0]),
                                    np.array([0, 0, zmax]),
                                    back_color)
        right_wall = create_wall_mesh(np.array([xmax, ymax, 0]),
                                    np.array([0, -ymax, 0]),
                                    np.array([0, 0, zmax]),
                                    right_color)
        left_wall = create_wall_mesh(np.array([xmin, ymax, 0]),
                                    np.array([0, -ymax, 0]),
                                    np.array([0, 0, zmax]),
                                    left_color)
        ceiling = create_wall_mesh(np.array([xmin, ymax, zmax]),
                                np.array([xmax - xmin, 0, 0]),
                                np.array([0, -ymax, 0]),
                                ceiling_color)
        
        # Add walls to objects list
        objects.extend([back_wall, right_wall, left_wall, ceiling])

    # Add the main objects

    for obj_data in object_positions:
        obj_file = obj_data['obj_file']
        xcoord = obj_data['xcoord']
        ycoord = obj_data['ycoord']
        w = obj_data['w']
        v1 = np.array([xcoord, ycoord, 0])
        u = np.array([1, 0, 0])
        theta = -np.clip(np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1)), -1, 1)
    
        # Load the object
        obj_path = os.path.join(object_folder, obj_file)
        obj = trimesh.load(obj_path, force='mesh')
        obj_extents = obj.extents
        scale_factors = [w / obj_extents[0], 1.1 / obj_extents[2]]  # Height is fixed at 1.1
        scale_factor = min(scale_factors) * 2
        obj.apply_scale(scale_factor)
        
        # Apply rotations and translations
        rotation = rotation_matrix(np.radians(90), [1, 0, 0])
        obj.apply_transform(rotation)
        rotation_z = rotation_matrix(theta, [0, 0, 1])
        obj.apply_transform(rotation_z)
        z_min = obj.vertices[:, 2].min()
        obj.apply_translation([0,0, -z_min])
        obj.apply_translation(v1)
        obj.visual.vertex_colors = object_color 
    
        scene_objects.append(obj)

    objects.extend(scene_objects)
    # Create the scene and add all meshes
    scene = trimesh.Scene()
    scene.add_geometry(objects)
    front_rotation = rotation_matrix(np.radians(90), [1, 0, 0])
    second_rotation = rotation_matrix(np.radians(180), [0, 0, 1])
    scene.apply_transform(front_rotation)
    scene.apply_transform(second_rotation) 
        
    # Add the laser position as a sphere
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
    #rotation = rotation_matrix(np.radians(90), [1, 0, 0])
    for pos in cam_pos:
        sphere = trimesh.creation.icosphere(radius=sphere_radius)
        sphere.apply_translation(pos)
        #sphere.apply_transform(rotation)
        sphere.front_wall_color = [200, 200, 200, 100]  
        sphere.visual.vertex_colors = camera_pixel_color
        camera_pixel.append(sphere)
        
    # Agregar todas las esferas a la escena
    scene.add_geometry(camera_pixel)
    
    ########################################################## Front wall
    # Define the wall dimensions and position
    x_start = xmin        # Start at x = 0
    x_end = 0        # End at x = xmax
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
    sphere_radius = 0.03  # Adjust the radius as needed
    wall_spheres = []
    #rotation_wall = rotation_matrix(np.radians(-90), [1, 0, 0])
    
    for pos in wall_points:
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
        sphere.apply_translation(pos)
        #sphere.apply_transform(rotation_wall)
        sphere.visual.vertex_colors = [200, 200, 200, 255]  # Color of the spheres
        wall_spheres.append(sphere)

    # Add the spheres representing the wall to the scene
    scene.add_geometry(wall_spheres)
    
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
    
    #### Scene customization for streamlit interface deplo
    # Create the 3D figure using Plotly
    fig3d = go.Figure()
    
    
    # Add each object to the plot
    for mesh in objects:
        vertices = mesh.vertices
        faces = mesh.faces
        x, y, z = vertices.T
        i, j, k = faces.T

        fig3d.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='blue',
            opacity=1.0,
            name='Walls & Ceiling'
        ))

    for idx, obj in enumerate(scene_objects):
    # Add the main object as a separate mesh
        fig3d.add_trace(go.Mesh3d(
            x=obj.vertices[:,0],
            y=obj.vertices[:,1],
            z=obj.vertices[:,2],
            i=obj.faces[:,0],
            j=obj.faces[:,1],
            k=obj.faces[:,2],
            color='orange',
            opacity=1.0,
            name=f'Object {idx + 1}'
        ))

    # Add the camera pixels as small green spheres
    for sphere in camera_pixel:
        x, y, z = sphere.vertices.T
        i, j, k = sphere.faces.T
        fig3d.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='green',
            opacity=1.0,
            name='Camera Pixels',
            showscale=False
        ))

    # Add the laser sphere as a red sphere
    fig3d.add_trace(go.Mesh3d(
        x=laser_sphere.vertices[:,0],
        y=laser_sphere.vertices[:,1],
        z=laser_sphere.vertices[:,2],
        i=laser_sphere.faces[:,0],
        j=laser_sphere.faces[:,1],
        k=laser_sphere.faces[:,2],
        color='red',
        opacity=1.0,
        name='Laser',
        showscale=False
    ))

    # Add the front wall spheres as semi-transparent grey spheres
    front_wall_points = wall_points
    fig3d.add_trace(go.Scatter3d(
        x=front_wall_points[:,0],
        y=front_wall_points[:,1],
        z=front_wall_points[:,2],
        mode='markers',
        marker=dict(
            size=2,
            color='white',
            opacity=1.0 
        ),
        name='Front Wall',
        showlegend=False
))

    # Update the layout for better visualization
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Scene Visualization",
        width=800,
        height=800
    )

    # Guardar los datos
    filename = f"Simulacion_Con_Malla_{int(params['bin_size'] * 1e12)}ps.mat"
    #savemat(filename, {'params': params, 'y_meas_vec': y_meas_vec_noisy_reshaped, 'objects': objects})
    return fig3d, y_meas_vec


# Streamlit App
def main():
    st.set_page_config(
        page_icon="img/logo.webp",
        layout="wide",
        initial_sidebar_state="auto",
    )
    
    def img_to_base64(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            logging.error(f"Error converting image to base64: {str(e)}")
            return None
    
    img_path = "img/logo.webp"
    img_base64 = img_to_base64(img_path)
    
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow" style="width:100%; height:auto; border-radius: 30px;">',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.write("Failed to load image.")
    
    st.title("Interactive Simulation for SPAD Camera in :red[_NLOS_] Scenarios :sunglasses:")


    # Initialize the sidebar parameters
    st.sidebar.header("Simulation Parameters")
    camera_FOV = st.sidebar.slider("Camera FOV", 0.1, 1.0, 0.25)
    cam_pixel_dim = st.sidebar.slider("Camera Pixel Dimension", 16, 64, 32, step=1)
    bin_size = st.sidebar.number_input("Bin Size (seconds)", value=390e-12, format="%.1e")
    laser_intensity = st.sidebar.slider("Laser Intensity (mW)", 50, 1000, 1000)
    obj_files = get_obj_files(object_folder)
    st.sidebar.subheader("Objects selection")

    selected_obj_files = st.sidebar.multiselect("Select 3D Objects", obj_files)
    object_positions = []
    for obj_file in selected_obj_files:
        with st.sidebar.expander(f"Position for {obj_file}"):
            xcoord = st.number_input(f"{obj_file} X Coordinate", value=0.0, key=f"x_{obj_file}")
            ycoord = st.number_input(f"{obj_file} Y Coordinate", value=1.25, key=f"y_{obj_file}")
            w = st.slider(f"{obj_file} Size", 0.1, 0.4, 0.2, key=f"w_{obj_file}")
            object_positions.append({'obj_file': obj_file, 'xcoord': xcoord, 'ycoord': ycoord, 'w': w})
            
    # Run overlap check
    overlaps = check_overlaps(object_positions)
    if overlaps:
        st.error("The following objects are overlapping. Please adjust their positions to avoid overlaps:")
        for obj1, obj2 in overlaps:
            st.write(f"- **{obj1}** overlaps with **{obj2}**")
        st.stop()  # Stop the app here to prevent the simulation from running

    hide_walls = st.sidebar.checkbox("Hide All Walls", value=False)

    # Run the simulation when button is clicked
    if st.sidebar.button("Run Simulation"):
        fig3d, y_meas_vec = simulation(camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls)

        # Store data in session state
        st.session_state['y_meas_vec'] = y_meas_vec
        st.session_state['fig3d'] = fig3d
        st.session_state['pixel_x'] = cam_pixel_dim // 2
        st.session_state['pixel_y'] = cam_pixel_dim // 2

    # Retrieve data from session state
    if 'y_meas_vec' in st.session_state and 'fig3d' in st.session_state:
        y_meas_vec = st.session_state['y_meas_vec']
        fig3d = st.session_state['fig3d']
    else:
        st.warning("Please run the simulation first.")
        return

    # Display the 3D scene
    st.plotly_chart(fig3d, use_container_width=True)

    # Display Simulation Results
    st.markdown("### Simulation Results")

    # Compute the Time-Integrated Intensity
    y_sum = np.sum(y_meas_vec, axis=2)
    y_sum = np.roll(y_sum, shift=1, axis=-1)

    # Create the Time-Integrated Intensity Plotly heatmap
    fig_intensity = go.Figure(data=go.Heatmap(
        z=y_sum,
        colorscale='Viridis',
        colorbar=dict(title='Intensity'),
    ))
    
    fig_intensity.update_layout(
        title='Time-Integrated Intensity',
        xaxis_title='Pixel X',
        yaxis_title='Pixel Y',
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Ensure square pixels
    )
        
    col1, col2 = st.columns(2)
    
    with col2:
        # Display the plot and capture click events
        selected_points = plotly_events(
            fig_intensity,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=400,
            override_width="100%"  
        )

    # Update selected pixel based on click
    if selected_points:
        point = selected_points[0]
        pixel_x = int(point['x'])
        pixel_y = int(point['y'])
        st.session_state['pixel_x'] = pixel_x
        st.session_state['pixel_y'] = pixel_y
    else:
        # Default pixel selection at the center if no click event
        pixel_x = st.session_state.get('pixel_x', cam_pixel_dim // 2)
        pixel_y = st.session_state.get('pixel_y', cam_pixel_dim // 2)

    # Get the temporal response for the selected pixel
    y_meas_vec_shifted = np.roll(y_meas_vec, shift=1, axis=1)
    temporal_response = y_meas_vec_shifted[pixel_y, pixel_x, :]

    # Create the Temporal Response Plotly line plot
    fig_temporal =  go.Figure(data=go.Scatter(
        y=temporal_response,
        mode='lines'
    ))
                                             
    fig_temporal.update_layout(
        title=f'Temporal Response at Pixel ({pixel_x}, {pixel_y})',
        xaxis_title='Time Bin',
        yaxis_title='Intensity'
    )
    
    with col1:
        st.plotly_chart(fig_temporal, use_container_width=True)

if __name__ == "__main__":
    main()