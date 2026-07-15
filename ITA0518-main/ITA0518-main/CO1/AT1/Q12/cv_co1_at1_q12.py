import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a perfect, straight-line grid (the real world)
grid = np.zeros((200, 200), dtype=np.uint8)  # Create a blank black image array filled with zeros
grid[::20, :] = 255
grid[:, ::20] = 255

# Step 2: Define a camera matrix and extreme barrel distortion coefficients
K = np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]], dtype=float)
D = np.array([-0.005, 0, 0, 0], dtype=float) # Negative k1 causes barrel effect

# Step 3: Mathematically simulate the light bending through the distorted lens
h, w = grid.shape
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w,h), 5)  # Generate undistortion maps from camera calibration
distorted = cv2.remap(grid, mapx, mapy, cv2.INTER_LINEAR)  # Apply geometric remap to correct lens distortion

# Step 4: Display the lens physics
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(grid, cmap='gray'), plt.title('Real World Geometry')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(distorted, cmap='gray'), plt.title('Lens Barrel Distortion')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
