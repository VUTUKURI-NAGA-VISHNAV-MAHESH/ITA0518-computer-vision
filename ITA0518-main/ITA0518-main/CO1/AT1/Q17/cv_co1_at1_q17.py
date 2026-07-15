import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a synthetic image with an object hidden in a shadow gradient
img = np.zeros((100, 200), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.circle(img, (100, 50), 30, 150, -1)  # Draw a filled or outlined circle on the image
gradient = np.tile(np.linspace(0.1, 1.0, 200), (100, 1))  # Repeat an array to build a larger image or gradient
shadow_img = (img * gradient).astype(np.uint8)

# Step 2: Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Create CLAHE object for adaptive contrast enhancement

# Step 3: Apply CLAHE to locally fix the contrast across the gradient
fixed_img = clahe.apply(shadow_img)  # Apply CLAHE to locally enhance image contrast

# Step 4: Visualize the correction
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(shadow_img, cmap='gray'), plt.title('Uneven Contrast')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(fixed_img, cmap='gray'), plt.title('CLAHE Restored')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
