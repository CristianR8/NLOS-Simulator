import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import imageio

def save_animation_as_gif(fig, filename, duration=300):
    # Create a list to store the frames
    frames = []
    
    # Iterate through each frame in the figure
    for frame in fig.frames:
        # Create a new figure for the current frame
        frame_fig = go.Figure(frame.data)
        frame_fig.update_layout(fig.layout)
        
        # Save the frame as an image
        frame_image = pio.to_image(frame_fig, format='png')
        
        # Append the image to the frames list
        frames.append(imageio.imread(frame_image))
    
    # Save the frames as a GIF
    imageio.mimsave(filename, frames, duration=duration/1000)  # duration is in seconds
    
def per_frame_heatmap(y_meas_vec, cam_pixel_dim, num_bins, bin_size, adjusted_x, adjusted_y):
    # No se calcula la suma acumulativa, se usan los datos directamente
    data = y_meas_vec  # Usamos los datos originales en lugar de la suma acumulativa
    
    # Decide el número de frames para la animación
    max_frames = 300
    step = max(1, num_bins // max_frames)
    
    # Crear los frames para la animación
    frames = []
    for t in range(0, num_bins, step):
        z = data[:, :, t]  # Tomamos el frame actual sin acumulación
        frame = go.Frame(
            data=go.Heatmap(
                x=adjusted_x,
                y=adjusted_y,
                z=z,
                colorscale='Viridis',
            ),
            name=f'Frame {t}',
        )
        frames.append(frame)
    
    # Datos iniciales (primer frame)
    initial_z = data[:, :, 0]
    fig = go.Figure(
        data=go.Heatmap(
            x=adjusted_x,
            y=adjusted_y,
            z=initial_z,
            colorscale='Viridis',
        ),
        frames=frames,
        layout=go.Layout(
            width=700,   
            height=700,   
            xaxis_title='Píxel X',
            yaxis_title='Píxel Y',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
        )
    )
    save_animation_as_gif(fig, 'per_frame_heatmap.gif')
    
    return fig

def create_animated_heatmap(y_meas_vec, cam_pixel_dim, num_bins, bin_size, adjusted_x, adjusted_y):
    # Calculate the cumulative sum over time.
    cumulative_sum = np.cumsum(y_meas_vec, axis=2)
    
    # Decide the number of frames for the animation 
    max_frames = 300
    step = max(1, num_bins // max_frames)
    
    # Create the frames for the animation
    frames = []
    for t in range(0, num_bins, step):
        z = cumulative_sum[:, :, t]
        frame = go.Frame(
            data=go.Heatmap(
                x=adjusted_x,
                y=adjusted_y,
                z=z,
                colorscale='Viridis',
                colorbar=dict(title='Intensidad', tickfont=dict(color='black'), titlefont=dict(color='black')),
            ),
            name=f'Frame {t}'
        )
        frames.append(frame)
    
    # Initial data
    initial_z = cumulative_sum[:, :, 0]
    fig = go.Figure(
        data=go.Heatmap(
            x=adjusted_x,
            y=adjusted_y,
            z=initial_z,
            colorscale='Viridis',
        ),
        frames=frames,
        layout=go.Layout(
            width=700,   
            height=700,   
            xaxis_title='Píxel X',
            yaxis_title='Píxel Y',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
        )
    )
    
    save_animation_as_gif(fig, 'cumulative.gif')
    
    return fig

def plot_temporal_response(time_axis, temporal_response, animation_speed, pixel_x=0, pixel_y=0, origin_x=0, origin_y=0, line_color='blue', marker_color='red', marker_size=10):

    
    # Create figure with initial traces
    fig_temporal = go.Figure(
        data=[
            # Line trace
            go.Scatter(
                x=time_axis,
                y=temporal_response,
                mode='lines',
                line=dict(width=2, color=line_color),
                name='Respuesta temporal'
            ),
            # Initial marker position
            go.Scatter(
                x=[time_axis[0]],
                y=[temporal_response[0]],
                mode='markers',
                marker=dict(color=marker_color, size=marker_size),
                name='Marcador'
            )
        ]
    )

    # Update layout
    fig_temporal.update_layout(
        title=f'Respuesta temporal en el píxel ({pixel_x - origin_x}, {pixel_y - origin_y})',
        xaxis_title='Intervalo de tiempo',
        yaxis_title='Intensidad',
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(
                args=[None, {
                    "frame": {"duration": animation_speed, "redraw": False},
                    "fromcurrent": True,
                    "transition": {"duration": animation_speed}
                }],
                label='Play',
                method='animate'
            )]
        )]
    )

    # Create animation frames
    frames = [
        go.Frame(
            data=[
                # Static line trace
                go.Scatter(x=time_axis, y=temporal_response, mode='lines'),
                # Moving marker
                go.Scatter(
                    x=[time_axis[k]],
                    y=[temporal_response[k]],
                    mode='markers',
                    marker=dict(color=marker_color, size=marker_size)
                )
            ],
            traces=[0, 1],
            name=f'frame{k}'
        )
        for k in range(len(time_axis))
    ]

    # Add frames to the figure
    fig_temporal.frames = frames
    
    save_animation_as_gif(fig_temporal, "temporal_response.gif")
    
    return fig_temporal