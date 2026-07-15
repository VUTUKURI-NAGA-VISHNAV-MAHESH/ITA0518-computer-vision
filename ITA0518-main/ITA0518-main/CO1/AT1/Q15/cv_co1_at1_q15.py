import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a white page document with dark text
page = np.full((100, 200), 255, dtype=np.uint8)  # Create a solid gray image array filled with a value
cv2.putText(page, 'OCR TEST', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # Render text characters onto the image matrix

# Step 2: Add a synthetic gradient shadow simulating bad illumination
gradient = np.tile(np.linspace(1.0, 0.1, 200), (100, 1))  # Repeat an array to build a larger image or gradient
shadow_page = (page * gradient).astype(np.uint8)

# Step 3: Use Adaptive Thresholding (1-bit Quantization optimization)
# This calculates thresholds locally, completely erasing the shadow gradient!
binary = cv2.adaptiveThreshold(shadow_page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)  # Apply adaptive local thresholding for binarization

# Step 4: Display the OCR prep step
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(shadow_page, cmap='gray'), plt.title('Poor Illumination')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(binary, cmap='gray'), plt.title('Adaptive Thresholded')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
