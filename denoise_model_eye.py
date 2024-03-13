import cv2
import numpy as np
import glob
import os
import json

spacing = 23.4742
mark_thickness = 1

data_path = r"D:\Projects\Dataset\temp_issue"
result_path = r"D:\Projects\Dataset\temp_issue\denoise"
os.makedirs(result_path, exist_ok=True)

img_list = glob.glob(f'{data_path}/*.png')

for img_path in img_list:
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print(src.shape[1])
    center_x = int(src.shape[1]/2)-1
    center_y = int(src.shape[0]/2)+1
    cv2.circle(src, (center_x, center_y), radius=0, color=(0, 0, 255), thickness=mark_thickness)

    # Define the kernel for averaging
    kernel = np.ones((3, 3), np.float32) / 9

    # Filter the image using the average filter
    filtered = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # Perform adaptive thresholding
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 2)

    # Create a spherical structuring element with a radius of 3
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Perform morphological opening
    binary_remove_noise = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)

    src = np.where(binary_remove_noise==255, src, 0)

    base_name = os.path.basename(img_path).split('.png')[0]
    cv2.imwrite(f'{result_path}/{base_name}-denoise.png', src)
