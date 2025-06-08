import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the dimensions for a half tennis court and extra space
court_length = 23.77 / 2  # Half of the full length
court_width = 8.23
extra_space = 4
service_box_length = court_length / 2

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Set axis limits to include extra space
total_length = court_length + extra_space * 2
total_width = court_width + extra_space * 2
ax.set_xlim(0, total_length)
ax.set_ylim(0, total_width)
ax.axis('off')

# Draw court lines (only once)
court_start_x = extra_space
court_start_y = extra_space
plt.plot([court_start_x, court_start_x], [court_start_y, court_start_y + court_width], color="black")  # Left sideline
plt.plot([court_start_x, court_start_x + court_length], [court_start_y + court_width, court_start_y + court_width], color="black")  # Baseline
plt.plot([court_start_x + court_length, court_start_x + court_length], [court_start_y + court_width, court_start_y], color="black")  # Right sideline
plt.plot([court_start_x + court_length, court_start_x], [court_start_y, court_start_y], color="black")  # Net line
plt.plot([court_start_x, court_start_x + court_length], [court_start_y + court_width, court_start_y + court_width], color="black")
plt.plot([court_start_x + service_box_length, court_start_x + service_box_length], [court_start_y, court_start_y + court_width], color="black")  # Center service line
plt.plot([court_start_x+service_box_length, court_start_x + service_box_length*2], [court_start_y + court_width/2 , court_start_y + court_width/2], color="black")  # Service line

# Load data points from CSV
csv_file = 'C:/Users/Buse/Desktop/arayuz_icin - 240424-1112_T016S3FE-A-001/DusmeNoktasi.txt'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)
points = list(zip((data['x_line'] / 100) + extra_space, (data['y_line'] / 100) + extra_space))

# Empty list to store processed points
processed_points = []

# Histogram parameters
x_edges = np.linspace(0, total_length, 50)
y_edges = np.linspace(0, total_width, 50)

# Initial plot with first frame to set up colorbar
heatmap, xedges, yedges = np.histogram2d([], [], bins=[x_edges, y_edges])
"""'viridis': Mavi-yeşil-sarı, genellikle önerilen default colormap'tir.
'plasma': Daha sıcak tonlar (mor-turuncu-sarı).
'inferno': Siyah-turuncu-sarı, daha yüksek kontrast sunar.
'magma': Siyah-beyaz-mor, yüksek kontrastlı koyu tonlar.
'cividis': Mavi-sarı, renk körü dostu bir skala.
'hot': Siyah-kırmızı-sarı-beyaz, sıcaklık haritaları için yaygın.
'coolwarm': Mavi-kırmızı, negatif-pozitif değerli veriler için uygundur."""
cax = ax.imshow(heatmap.T, extent=[x_edges[0], x_edges[-1], y_edges[0], yedges[-1]], origin='lower', cmap='magma', aspect='auto', alpha=0.7)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Number of Points')

# Iterate through each point and update the heatmap
for i, point in enumerate(points):
    # Add the current point to the list of processed points
    processed_points.append(point)

    # Convert processed points to numpy array
    x, y = np.array(processed_points).T

    # Update the histogram as a heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    # Update the data of the heatmap (no need to clear the plot)
    cax.set_data(heatmap.T)

    # Rescale color limits based on current point density
    cax.set_clim(0, np.max(heatmap))

    # Update title with the current point index
    plt.title(f"Tennis Court Heatmap with Extra Space - Point {i + 1}/{len(points)}")

    # Display the updated plot
    plt.pause(0.2)  # Pause to visualize each step

# Show the final plot
plt.show()
