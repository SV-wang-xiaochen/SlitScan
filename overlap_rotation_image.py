import cv2
import numpy as np
import glob

data_path = r"D:\Projects\Dataset\issue_137\eye_0314\LY\5"
img_list = glob.glob(f'{data_path}/*.jpg')

# Load the images
image1 = cv2.imread(img_list[0])
image2 = cv2.imread(img_list[1])
image3 = cv2.imread(img_list[2])

# Ensure both images have the same dimensions
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Create an output image with the same size as the input images
output = np.zeros_like(image1)

# # Assign different colors for each image
# output[:, :, 0] = image2[:, :, 0]  # Blue channel for image1
# output[:, :, 1] = image1[:, :, 1]  # Green channel for image1
# output[:, :, 2] = image2[:, :, 2]  # Red channel for image2

output = np.maximum(np.maximum(image1, image2), image3)

# output = np.maximum(image2, image3)

# # Display the result
# cv2.imshow('Overlapped Images', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('all.jpg', output)