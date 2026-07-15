import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create a clean base image
img = np.full((100, 100), 128, dtype=np.uint8)  # Create a solid gray image array filled with a value

# Step 2: Introduce Synthetic Salt-and-Pepper sensor noise
noise = np.random.randint(0, 100, (100, 100))  # Generate random numbers to simulate sensor noise
img[noise > 95] = 255 # Salt (White)
img[noise < 5] = 0    # Pepper (Black)

# Step 3: Clean the noise using a Median Filter
# The '5' indicates a 5x5 pixel neighborhood used to calculate the mathematical median
clean_img = cv2.medianBlur(img, 5)  # Apply median filter to remove salt-and-pepper noise

# Step 4: Display the cleaned image
plt.figure(figsize=(8,4))  # Create a new figure window for display
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Sensor Noise')  # Place this image in a subplot grid position
plt.subplot(122), plt.imshow(clean_img, cmap='gray'), plt.title('Median Filter Applied')  # Place this image in a subplot grid position
plt.show()  # Show the final plot window to the user
