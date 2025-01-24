import numpy as np
import plotly.graph_objects as go

def create_animated_heatmap(y_meas_vec, cam_pixel_dim, num_bins, bin_size, adjusted_x, adjusted_y):
    # Calculate the cumulative sum over time.
    cumulative_sum = np.cumsum(y_meas_vec, axis=2)
    
    # Decide the number of frames for the animation 
    max_frames = 100
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
            colorbar=dict(title='Intensidad', tickfont=dict(color='black'), titlefont=dict(color='black')),
        ),
        frames=frames,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True},
                                         "fromcurrent": True, "transition": {"duration": 0}}]
                        )
                    ],
                    showactive=False,
                    x=0.5,  
                    y=0,   
                    xanchor="center",
                    yanchor="top"
                )
            ],
            width=700,   
            height=600,   
            xaxis_title='Píxel X',
            yaxis_title='Píxel Y',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
        )
    )
    
    return fig
