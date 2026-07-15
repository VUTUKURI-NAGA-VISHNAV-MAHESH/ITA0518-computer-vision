import cv2  # Import OpenCV for image processing algorithms
import numpy as np                                           # Import Numpy for mathematical matrix operations
import matplotlib.pyplot as plt                              # Import Matplotlib to visualize the output on screen

# Step 1: Create two sequential frames simulating an object moving to the right
frame1 = np.zeros((100, 100), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.rectangle(frame1, (20, 20), (40, 40), 255, -1)  # Draw a filled or outlined rectangle on the image

frame2 = np.zeros((100, 100), dtype=np.uint8)  # Create a blank black image array filled with zeros
cv2.rectangle(frame2, (40, 20), (60, 40), 255, -1) # Object moved +20 pixels

# Step 2: Compute Dense Optical Flow (A core surveillance algorithm)
# This calculates the velocity vector of every pixel between the two frames
flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # Compute dense optical flow between two frames
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Convert Cartesian (x,y) vectors to polar (mag, angle)

# Step 3: Visualize the motion magnitude
plt.imshow(mag, cmap='hot')  # Render the image array to the display
plt.title('Optical Flow Magnitude (Motion Tracking)')  # Add a descriptive title to the plot
plt.colorbar()  # Add a color legend bar to the plot
plt.show()  # Show the final plot window to the user
