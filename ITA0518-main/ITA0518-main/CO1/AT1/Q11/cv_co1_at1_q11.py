import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a clean base image
img = np.full((100, 100), 100, dtype=np.uint8)  # Create a solid gray image array filled with a value

# Step 2: Simulate a high-end CCD sensor (Virtually zero read noise)
ccd_img = img.copy()

# Step 3: Simulate an older/cheap CMOS sensor (High read noise and thermal noise)
noise = np.random.normal(0, 15, img.shape).astype(np.int16)  # Generate random numbers to simulate sensor noise
cmos_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)  # Clamp all values into a valid pixel range (0-255)

# Step 4: Compare the acquisitions side by side
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(ccd_img, cmap='gray'), plt.title('Simulated CCD (Clean)')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(cmos_img, cmap='gray'), plt.title('Simulated CMOS (Noisy)')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
