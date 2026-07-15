import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create an image with complex, smooth lighting
img = np.zeros((100, 100), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.circle(img, (50, 50), 50, 255, -1)  # Draw a filled or outlined circle on the image
img = cv2.GaussianBlur(img, (49, 49), 20) # Soft blur gradient

# Step 2: Reduce quantization from 256 levels down to just 4 levels (2-bit)
# This forces every pixel to snap to one of 4 rigid gray values
quantized_4_level = np.floor(img / 64) * 64  # Round down each pixel value to quantize intensity

# Step 3: Display the impact of poor quantization
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('256-Level Quantization')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(quantized_4_level, cmap='gray'), plt.title('4-Level Quantization')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
