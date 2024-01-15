import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('optical_origin.jpg')
image2 = cv2.imread('optical_correction.jpg')

# Ensure both images have the same dimensions
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Create an output image with the same size as the input images
output = np.zeros_like(image1)

# Assign different colors for each image
output[:, :, 0] = image2[:, :, 0]  # Blue channel for image1
output[:, :, 1] = image1[:, :, 1]  # Green channel for image1
output[:, :, 2] = image2[:, :, 2]  # Red channel for image2

# # Display the result
# cv2.imshow('Overlapped Images', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('optical_overlap.jpg', output)