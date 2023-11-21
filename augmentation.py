import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

path = r'..\Dataset\DigitalGain'
image_list = glob.glob(f'{path}/**/*.bmp')
image_index = 0

save_folder = './results'
os.makedirs(save_folder, exist_ok = True)

for item in image_list:
    print(item)

    base_name = os.path.basename(item).split('.')[0]
    digital_gain_factor = item.split('\\')[3]

    # Read the BMP image
    image = cv2.imread(item, cv2.IMREAD_COLOR)

    # Check if the image is successfully loaded
    if image is None:
        print("Error: Could not read the image.")
        exit()

    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    # print(img_yuv[0].shape)
    equalized_V = cv2.equalizeHist(img_HSV[:,:,2])
    img_HSV[:,:,2] = equalized_V

    equalized_image = cv2.cvtColor(img_HSV,cv2.COLOR_HSV2BGR)

    # Convert the image to float32
    image_float = equalized_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma = 1.5  # You can adjust the gamma value as needed
    gamma_corrected = np.power(image_float, gamma)

    # Convert back to uint8
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    # # Convert the image to float32
    # image_float = image.astype(np.float32) / 255.0

    # # Convert back to uint8
    # gamma_corrected = gammaCorrection(image, 0.8)

    # Display the original and equalized images using matplotlib
    # plt.figure(figsize=(10, 5))
    my_dpi = 20
    plt.figure(figsize=(4608 / my_dpi, 4544 / my_dpi), dpi=my_dpi)

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.savefig(f'{save_folder}/{digital_gain_factor}_{base_name}.png')

    # plt.show()
