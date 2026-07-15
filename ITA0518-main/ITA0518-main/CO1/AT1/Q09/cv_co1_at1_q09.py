import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a synthetic image representing a human face layout
img = np.zeros((100, 100), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.circle(img, (50, 50), 40, 255, -1) # Face base

# Step 2: Simulate Mid-Level Feature Extraction (Edge Detection)
# Feature extraction is a vital mid-level step before passing data to an AI model
edges = cv2.Canny(img, threshold1=100, threshold2=200)  # Detect edges using the Canny edge detection algorithm

# Step 3: Visualize the Mid-Level output
plt.imshow(edges, cmap='gray')  # Render the image array to the display
plt.title('Mid-Level Feature Extraction (Topology Outline)')  # Add a descriptive title to the plot
plt.show()  # Show the final plot window to the user
