import plotly.graph_objects as go
import cv2
import numpy as np
from plotly.subplots import make_subplots

from ImprovedBilateralFilter import BlurMetrics

# Load a grayscale image
gray_image = cv2.imread('filepath here', cv2.IMREAD_GRAYSCALE)

# Define the bottom and top values for spatial size and range size
spatial_size_bottom = 0
spatial_size_top = 40
range_size_bottom = 0
range_size_top = 40

# Calculate the range sizes
spatial_size_range = spatial_size_top - spatial_size_bottom + 1
range_size_range = range_size_top - range_size_bottom + 1

# Initialize the blur levels array
blur_levels_frequency = np.zeros((spatial_size_range, range_size_range))
blur_levels_laplacian = np.zeros((spatial_size_range, range_size_range))
blur_levels_gradient = np.zeros((spatial_size_range, range_size_range))
blur_levels_brenner = np.zeros((spatial_size_range, range_size_range))
images = []

# Loop through spatial and range sizes
for i, spatial_size in enumerate(range(spatial_size_bottom, spatial_size_top + 1)):
    print('Spatial Size: ' + str(spatial_size))
    for j, range_size in enumerate(range(range_size_bottom, range_size_top + 1)):
        print('Range Size: ' + str(range_size))
        current_filtered_image = cv2.bilateralFilter(gray_image, 0, spatial_size, range_size)
        images.append(current_filtered_image)
        blur_levels_frequency[i, j] = BlurMetrics.measure_blur(current_filtered_image, "local_fourier_analysis")
        blur_levels_laplacian[i, j] = BlurMetrics.measure_blur(current_filtered_image, "laplacian_variance")
        blur_levels_gradient[i, j] = BlurMetrics.measure_blur(current_filtered_image, "gradient_magnitude")
        blur_levels_brenner[i, j] = BlurMetrics.measure_blur(current_filtered_image, "brenner_focus_measure")

np_images = np.array(images)
np.save("images.npy", np_images)

blur_levels_frequency /= np.linalg.norm(blur_levels_frequency)
blur_levels_laplacian /= np.linalg.norm((blur_levels_laplacian))
blur_levels_gradient /= np.linalg.norm((blur_levels_gradient))
blur_levels_brenner /= np.linalg.norm((blur_levels_brenner))

blur_levels_brenner = np.flipud(blur_levels_brenner)
blur_levels_laplacian = np.flipud(blur_levels_laplacian)
blur_levels_gradient = np.flipud(blur_levels_gradient)

trace_frequency = go.Contour(z=blur_levels_frequency, name='Frequency')
trace_laplacian = go.Contour(z=blur_levels_laplacian, name='Laplacian')
trace_gradient = go.Contour(z=blur_levels_gradient, name='Gradient')
trace_brenner = go.Contour(z=blur_levels_brenner, name='Brenner')

fig = make_subplots(rows=1, cols=1)

# Add traces to the plot
fig.add_trace(trace_frequency)
fig.add_trace(trace_laplacian)
fig.add_trace(trace_gradient)
fig.add_trace(trace_brenner)

# Update layout to make surfaces overlaid
fig.update_layout(scene=dict(
                    xaxis=dict(title='Spatial size'),
                    yaxis=dict(title='Range size'),
                    zaxis=dict(title='Blur level')),
                  title='Surface Overlay Plot')

# Add buttons for toggling each trace
buttons = []
for trace in [trace_frequency, trace_laplacian, trace_gradient, trace_brenner]:
    button = dict(label=trace.name,
                  method='update',
                  args=[{'visible': [True if t == trace else False for t in fig.data]},
                        {'title': 'Surface Overlay Plot - {}'.format(trace.name)}])
    buttons.append(button)

fig.update_layout(updatemenus=[dict(type="buttons", direction="right", buttons=buttons)])

fig.show()