import cv2
import numpy as np

# Load the two images
src = cv2.imread(r'D:\Projects\Dataset\AngioAnnotation\labeled\00101_20240229008_00101_20220523001_OD_2022-05-23_10-32-15Angio 12x12 1024x1024 R2_27.6244_cSSO_2024-02-29_14-35-41.png')
dstRef = cv2.imread(r'D:\Projects\Dataset\AngioAnnotation\labeled\00101_20240229008_00101_20220523001_OD_2022-05-23_10-32-15Angio 12x12 1024x1024 R2_27.6244_EnhancedAngio_Retina.png')

srcTri = np.array([[195, 913], [985, 251], [147, 28]]).astype(np.float32)
dstTri = np.array([[197, 928], [967, 261], [139, 43]]).astype(np.float32)

warp_mat = cv2.getAffineTransform(srcTri, dstTri)

dstWarp = cv2.warpAffine(src, warp_mat, (dstRef.shape[1], dstRef.shape[0]))

# # Ensure both images have the same dimensions
# image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
#
# Create an output image with the same size as the input images
output = np.zeros_like(dstRef)

# Assign different colors for each image
# output[:, :, 0] = dstWarp[:, :, 0]
output[:, :, 1] = dstWarp[:, :, 1]
output[:, :, 2] = dstRef[:, :, 2]

# # Display the result
# cv2.imshow('Overlapped Images', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('alignment.jpg', output)