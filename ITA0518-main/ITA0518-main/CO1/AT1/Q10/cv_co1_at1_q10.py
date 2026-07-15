import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create an image with a clear, readable letter 'A'
img = np.zeros((100, 100), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.putText(img, 'A', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)  # Render text characters onto the image matrix

# Step 2: Simulate an extremely low sampling rate (sensor with few pixels)
low_res = cv2.resize(img, (10, 10))  # Resize image to new dimensions using interpolation

# Step 3: Attempt to resize it back up using interpolation to 'enhance' it
# This proves that lost sampling data cannot simply be interpolated back
upscaled = cv2.resize(low_res, (100, 100), interpolation=cv2.INTER_LINEAR)  # Resize image to new dimensions using interpolation

# Step 4: Show the permanent loss of visibility
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('High Sampling')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(upscaled, cmap='gray'), plt.title('Low Sampling (Data Lost)')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
