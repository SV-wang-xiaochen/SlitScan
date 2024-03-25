import os.path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

mark_thickness = 2
crop = 50

data_path = "./strips"
result_path = "./strips/results"

os.makedirs(result_path, exist_ok = True)
img_list = glob.glob(f'{data_path}/*.bmp')

for img_path in img_list:
    strip_color = cv2.imread(img_path)
    strip = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    row, col = strip.shape
    # half_max_values = np.max(strip, axis=0) / 2
    # half_max_values = np.reshape(half_max_values, (1, 4608))
    # half_max_values = np.tile(half_max_values, (row, 1))

    strip[:crop, :] = 0
    strip[-crop:,:] = 0

    above_half_max = np.where(strip>50, 1, 0)

    # _, binary = cv2.threshold(above_half_max,1,254, cv2.THRESH_BINARY)

    # Create a spherical structuring element with a radius of 5
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Perform morphological closing
    mask = cv2.morphologyEx(np.uint8(above_half_max), cv2.MORPH_CLOSE, se)

    mask[:crop, :] = 0
    mask[-crop:,:] = 0

    base_name = os.path.basename(img_path).split('.')[0]
    # # print(f'{result_path}/{base_name}-above-half.png')
    cv2.imwrite(f'{result_path}/{base_name}-above-50-close-crop.png', mask*255)

    increment = np.arange(row)
    increment = np.reshape(increment, (row,1))
    increment = np.tile(increment, (1, col))
    increment[:crop, :] = 0
    increment[-crop:, :] = 0

    # increment_masked = np.multiply(increment, mask)
    # mid_curve = np.sum(increment_masked, 0)/np.sum(mask, 0)

    # height of bright region
    bright_region_height = np.sum(mask, 0)

    # average intensity of bright region
    aver_intensity = np.divide(np.sum(np.multiply(strip, mask), 0), np.sum(mask, 0)+0.001)

    # middle point curve of bright region
    increment_masked_for_max = np.multiply(increment, mask)
    max_curve = np.max(increment_masked_for_max, 0)

    increment_masked_for_min = np.where(increment_masked_for_max == 0, 256, increment_masked_for_max)
    min_curve = np.min(increment_masked_for_min, 0)

    mid_curve = (max_curve + min_curve) / 2

    # Plotting the curve
    plt.plot(mid_curve)
    ax = plt.gca()  # you first need to get the axis handle
    ax.set_aspect(3)  # sets the height to width ratio to 1.5.
    plt.ylim(50, 150)
    plt.tight_layout()
    plt.show()

    # plot middle point curve of bright region
    for x, y in enumerate(mid_curve):
        cv2.circle(strip_color, (x, int(y)), radius=1, color=(0, 255, 0), thickness=mark_thickness)
    cv2.imwrite(f'{result_path}/{base_name}-mid-curve.png', strip_color)

