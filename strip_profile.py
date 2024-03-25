import os.path

import cv2
import numpy as np
import glob

data_path = "./strips"
result_path = "./strips/results"

os.makedirs(result_path, exist_ok = True)
img_list = glob.glob(f'{data_path}/*.bmp')

for img_path in img_list:
    strip = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    row, col = strip.shape
    half_max_values = np.max(strip, axis=0) / 2
    half_max_values = np.reshape(half_max_values, (1, 4608))
    half_max_values = np.tile(half_max_values, (row, 1))

    above_half_max = np.where(strip>half_max_values, 1, 0)
    print(above_half_max.shape)
    print(above_half_max)
    # _, binary = cv2.threshold(above_half_max,1,254, cv2.THRESH_BINARY)

    # Create a spherical structuring element with a radius of 5
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Perform morphological closing
    open_image = cv2.morphologyEx(np.uint8(above_half_max), cv2.MORPH_OPEN, se)

    base_name = os.path.basename(img_path).split('.')[0]
    # # print(f'{result_path}/{base_name}-above-half.png')
    cv2.imwrite(f'{result_path}/{base_name}-above-half-open.png', open_image*255)
