import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Simulate the acquisition of Red and Near-Infrared (NIR) satellite bands
red_band = np.full((100, 100), 50, dtype=np.float32)  # Create a solid gray image array filled with a value
nir_band = np.full((100, 100), 150, dtype=np.float32)  # Create a solid gray image array filled with a value

# Step 2: Introduce a "forest" area in the center (High NIR reflection, Low Red reflection)
cv2.rectangle(red_band, (30, 30), (70, 70), 20, -1)  # Draw a filled or outlined rectangle on the image
cv2.rectangle(nir_band, (30, 30), (70, 70), 220, -1)  # Draw a filled or outlined rectangle on the image

# Step 3: Calculate the Normalized Difference Vegetation Index (NDVI)
# Formula: (NIR - Red) / (NIR + Red)
ndvi = (nir_band - red_band) / (nir_band + red_band)  # Calculate Normalized Difference Vegetation Index

# Step 4: Display the vegetation map
plt.imshow(ndvi, cmap='RdYlGn')  # Render the image array to the display
plt.title('Calculated NDVI (Vegetation Index)')  # Add a descriptive title to the plot
plt.colorbar()  # Add a color legend bar to the plot
plt.show()  # Show the final plot window to the user
