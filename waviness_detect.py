import cv2
import numpy as np

# Load image
input_image = "/home/fahadabul/mask_rcnn_skyhub/waviness_final/8Fbhf2iE.jpeg"
image = cv2.imread(input_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Apply edge detection (Canny)
edges = cv2.Canny(blurred, 50, 150)

# Find contours from edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Save results instead of displaying
cv2.imwrite("/home/fahadabul/mask_rcnn_skyhub/waviness_final/edges_output.jpg", edges)
cv2.imwrite("/home/fahadabul/mask_rcnn_skyhub/waviness_final/contours_output.jpg", output)

print("Processing complete. Edge and contour images saved successfully!")

