import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

# set gamma here.
# 1 means no gamma correction.
gamma = 1.5

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

    # Get the dimensions of the image
    height, width, _ = image.shape

    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    equalized_V = cv2.equalizeHist(img_HSV[:,:,2])
    img_HSV[:,:,2] = equalized_V

    equalized_image = cv2.cvtColor(img_HSV,cv2.COLOR_HSV2BGR)

    # Convert the image to float32
    image_float = equalized_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(image_float, gamma)

    # Convert back to uint8
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    my_dpi = 20
    plt.figure(figsize=(4608*3/ my_dpi, 4544 / my_dpi), dpi=my_dpi)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'equalHist+gamma')
    plt.imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.savefig(f'{save_folder}/{digital_gain_factor}_{base_name}.png')

    # plt.show()
