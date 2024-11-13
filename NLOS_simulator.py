import streamlit as st
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from noise import add_sensor_noise, add_environmental_noise, add_shot_noise
import glob
import os
import base64
import logging
import plotly.graph_objects as go
from check_overlaps import check_overlaps
from streamlit_plotly_events import plotly_events


c = 299792458

object_folder = '/home/cristianr/NLOS-Simulator/objects/'

# Room Dimensions fixed 
ymin = 0
zmin = 0

# Function to list all .obj files in the folder
def get_obj_files(folder):
    return [os.path.basename(f) for f in glob.glob(os.path.join(folder, '*.obj'))]

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

def create_sparse_wall(origin, width_vec, height_vec, color, sphere_radius, spacing):
    """
    Crea una pared escasa compuesta por esferas distribuidas en una cuadrícula.
    
    :param origin: Vector de origen de la pared (esquina inferior izquierda).
    :param width_vec: Vector que define la anchura de la pared.
    :param height_vec: Vector que define la altura de la pared.
    :param color: Color de las esferas.
    :param sphere_radius: Radio de cada esfera.
    :param spacing: Espaciado entre las esferas.
    :return: Lista de esferas (Trimesh meshes).
    """
    wall_spheres = []
    
    # Calcular el número de esferas en cada dimensión
    num_x = int(np.linalg.norm(width_vec) // spacing)
    num_z = int(np.linalg.norm(height_vec) // spacing)
    
    # Vectores unitarios para iterar
    unit_width = width_vec / np.linalg.norm(width_vec)
    unit_height = height_vec / np.linalg.norm(height_vec)
    
    for i in range(num_x + 1):
        for j in range(num_z + 1):
            # Posición de cada esfera
            pos = origin + i * spacing * unit_width + j * spacing * unit_height
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
            sphere.apply_translation(pos)
            sphere.visual.vertex_colors = color
            wall_spheres.append(sphere)
    
    return wall_spheres

def simulation(xmin, xmax, ymax, zmax, camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls): 
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
    
    furthest_scene_point = np.array([xmax, ymax, zmax])
    furthest_spad_point = np.array([-params['camera_FOV'] / 2, -params['camera_FOV'], 0])
    d1 = np.linalg.norm(furthest_scene_point - laser_pos)
    d2 = np.linalg.norm(furthest_spad_point - furthest_scene_point)
    max_dist_travel = d1 + d2
    num_time_bins = int(np.ceil((max_dist_travel / c / bin_size + 0.2 * max_dist_travel / c / bin_size)))
    params['num_time_bins'] = num_time_bins
    
    
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
        
    x_start = xmin       
    x_end = 0       
    y_pos = ymin        
    z_start = zmin     
    z_end = zmax        
    
    sphere_radius = 0.015 
    spacing = 0.3          
    front_wall_color = [200, 200, 200, 255]  
    
    front_wall_origin = np.array([xmin, ymin, zmin])
    front_wall_width = np.array([x_end - x_start, 0, 0])  
    front_wall_height = np.array([0, 0, z_end - z_start])  
    
    front_wall_spheres = create_sparse_wall(
        origin=front_wall_origin,
        width_vec=front_wall_width,
        height_vec=front_wall_height,
        color=front_wall_color,
        sphere_radius=sphere_radius,
        spacing=spacing
    )
    
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
        scale_factor = min(scale_factors) 
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
    
    # Combinar las mallas y crear el intersector de rayos
    combined_mesh = trimesh.util.concatenate(objects)

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

    # Iterar sobre cada triángulo de la malla
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
    
    #### Scene customization for streamlit interface deploy
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

    cam_x = cam_pos[:, 0]
    cam_y = cam_pos[:, 1]
    cam_z = cam_pos[:, 2]

    fig3d.add_trace(go.Scatter3d(
        x=cam_x,
        y=cam_y,
        z=cam_z,
        mode='markers',
        marker=dict(
            size=3,  
            color='green',
            opacity=0.8
        ),
        name='Camera Pixels',
        showlegend=False
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
    
    # Añadir la pared frontal escasa como Scatter3d
    front_wall_x = []
    front_wall_y = []
    front_wall_z = []

    for sphere in front_wall_spheres:
        front_wall_x.extend(sphere.vertices[:, 0])
        front_wall_y.extend(sphere.vertices[:, 1])
        front_wall_z.extend(sphere.vertices[:, 2])

    fig3d.add_trace(go.Scatter3d(
        x=front_wall_x,
        y=front_wall_y,
        z=front_wall_z,
        mode='markers',
        marker=dict(
            size=3, 
            color='white',
            opacity=0.6
        ),
        name='Front Wall',
        showlegend=False
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1),  
                center=dict(x=0, y=0, z=0), 
                up=dict(x=0, y=0, z=1)  
            )
        ),
        title="3D Scene Visualization",
        width=800,
        height=800
    )
    
    # filename = f"Simulacion_Con_Malla_{int(params['bin_size'] * 1e12)}ps.mat"
    # savemat(filename, {'params': params, 'y_meas_vec': y_meas_vec_noisy_reshaped, 'objects': objects})
    
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
    st.sidebar.header("Simulation parameters", divider="red")
    
    # Room Dimensions
    st.sidebar.subheader("Room dimensions")
    xmin = st.sidebar.number_input("X Min", value=-1.5, format="%.1f")
    xmax = st.sidebar.number_input("X Max", value=1.5, format="%.1f")
    ymax = st.sidebar.number_input("Y Max", value=3.0, format="%.1f")
    zmax = st.sidebar.number_input("Z Max", value=3.0, format="%.1f")
    
    # Ensure valid room boundaries
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        st.error("Invalid room dimensions! Make sure max values are greater than min values.")
        st.stop()
        
    st.sidebar.subheader("Camera and laser parameters")

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
            w = st.slider(f"{obj_file} Size", 0.1, 5.0, 0.5, key=f"w_{obj_file}")
            object_positions.append({'obj_file': obj_file, 'xcoord': xcoord, 'ycoord': ycoord, 'w': w})
            
    # Object Size Validation
    exceeds = False
    for obj in object_positions:
        obj_file = obj['obj_file']
        x = obj['xcoord']
        y = obj['ycoord']
        w = obj['w']
        
        obj_path = os.path.join(object_folder, obj_file)
        obj = trimesh.load(obj_path, force='mesh')
        
        v1 = np.array([xcoord, ycoord, 0])
        u = np.array([1, 0, 0])
        theta = -np.clip(np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1)), -1, 1)
        
        # Calculate object boundaries based on its center and width
        min_x = x - w / 2
        max_x = x + w / 2
        min_y = y - w / 2
        max_y = y + w / 2
        min_z = obj.vertices[:, 2].min()
        max_z = min_z + w * 1.1  

        # Check if the object exceeds the room boundaries
        if min_x < xmin or max_x > xmax:
            st.error(f"**{obj_file}** exceeds the room boundaries along the X-axis.")
            exceeds = True
        if min_y < ymin or max_y > ymax:
            st.error(f"**{obj_file}** exceeds the room boundaries along the Y-axis.")
            exceeds = True
        if max_z > zmax:
            st.error(f"**{obj_file}** exceeds the room boundaries along the Z-axis.")
            exceeds = True
            
        if exceeds:
            st.error("Please adjust the size or position of the objects to fit within the room.")
            st.stop()  # Detener la ejecución para prevenir la simulación
        
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
        fig3d, y_meas_vec = simulation(xmin, xmax, ymax, zmax, camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls)

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
        colorbar=dict(title='Intensity', tickfont=dict(color='white'), titlefont=dict(color='white')),
        
    ))
    
    fig_intensity.update_layout(
        title='Time-Integrated Intensity',
        title_font=dict(color='white'),
        xaxis_title='Pixel X',
        xaxis=dict(titlefont=dict(color='white'), tickfont=dict(color='white')),
        yaxis_title='Pixel Y',
        yaxis=dict(titlefont=dict(color='white'), tickfont=dict(color='white'), scaleanchor="x", scaleratio=1),  # Ensure square pixels
        plot_bgcolor='#0e1017',
        paper_bgcolor='#0e1017',
    )
        
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        st.plotly_chart(fig_temporal, use_container_width=True)

if __name__ == "__main__":
    main()
