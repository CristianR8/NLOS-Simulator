import streamlit as st
import numpy as np

def get_photon_arrival_time(y_meas_vec, pixel_x, pixel_y, bin_size):
        # Get the temporal response for the selected pixel
        temporal_response = y_meas_vec[pixel_y, pixel_x, :]        
        # Find the rate of the maximum intensity in the time response
        measured_bin = np.argmax(temporal_response)
        measured_time = measured_bin * bin_size
        
        return measured_bin

def validate_simulation_photon_time(laser_pos, object_pos, c, measured_time, bin_size):
    # Calculate the Euclidean distance
    distance = np.linalg.norm(np.array(object_pos) - np.array(laser_pos))
    
    # Calculate Theoretical Time of Flight (ToF)
    tof_theoretical = 2 * distance / c  

    # Show results
    st.write(f"**Bin size:** {bin_size}")
    st.write(f"**Posicion del objeto:** {object_pos}")
    st.write(f"**Distancia Euclidiana (m):** {distance:.4f}")
    st.write(f"**Tiempo Teórico de Vuelo (s):** {tof_theoretical:.4e}")
    st.write(f"**Tiempo Registrado (s):** {measured_time:.6e}")
    st.write(f"**Dif (s):** {abs(tof_theoretical - measured_time)}")

    # Validation
    if abs(tof_theoretical - measured_time) <= bin_size:
        st.success("✅ La simulación es precisa: el tiempo registrado coincide con el teórico.")
    else:
        st.error("❌ La simulación no es precisa: el tiempo registrado no coincide con el teórico.")
        
### Usage example        

""" if 'y_meas_vec' in st.session_state:
    laser_pos = params['laser_pos']

    if len(object_positions) == 0:
        st.error("No hay objetos en la escena")
        st.stop()
    else:
        object = object_positions[0]
        object_pos = [object['xcoord'], object['ycoord'], object['zcoord']]

    pixel_x = st.session_state.get('pixel_x', cam_pixel_dim // 2)
    pixel_y = st.session_state.get('pixel_y', cam_pixel_dim // 2)

    bin_size = params['bin_size']
    c = params['c']

    # Obtener el tiempo de llegada del fotón registrado
    measured_time = get_photon_arrival_time(y_meas_vec_shifted, pixel_x, pixel_y, bin_size)

    # Validar la simulación
    validate_simulation_photon_time(laser_pos, object_pos, c, measured_time, bin_size) """