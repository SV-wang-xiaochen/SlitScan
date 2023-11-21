import cv2, glob
import numpy as np
import matplotlib.pyplot as plt


path = r'..\Dataset\DigitalGain'
image_list = glob.glob(f'{path}/**/*.bmp')
image_index = 5

for item in image_list:
    # print(item)
    # Read the image
    image = cv2.imread(item)

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create a black mask with the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the center and axes length of the ellipse
    center = (2200, 2300)
    axesLength = (2300, 2400)

    # Draw the circle on the mask
    cv2.ellipse(mask, center, axesLength, 0, 0, 360, (255, 255, 255), thickness=cv2.FILLED)

    # Use the mask to crop the image
    result_inner = cv2.bitwise_and(image, image, mask=mask)
    result_outer = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    gray_inner = cv2.cvtColor(result_inner, cv2.COLOR_BGR2GRAY)
    gray_outer = cv2.cvtColor(result_outer, cv2.COLOR_BGR2GRAY)

    print(f'Inner Grayscale Range: {gray_inner.min()}-{gray_inner.max()}')
    print(f'Outer Grayscale Range: {gray_outer.min()}-{gray_outer.max()}')
    print('\n')

    # # for debug
    # cv2.ellipse(image, center, axesLength,0, 0, 360, (0, 255, 0), thickness=10)
    # cv2.circle(image, center, 1, (0, 255, 0), thickness=50)

    my_dpi = 20
    plt.figure(figsize=(4608 * 2 / my_dpi, 4544 / my_dpi), dpi=my_dpi)
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(cv2.cvtColor(result_inner, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # plt.savefig(f'1.png')