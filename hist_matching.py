import cv2
import numpy as np


def hist_matching(image, reference):
    hist_image = cv2.calcHist([image], [0], None, [256], [0,256])
    hist_reference = cv2.calcHist([reference], [0], None, [256], [0,256])

    cdf_image = hist_image.cumsum()/hist_image.sum()
    cdf_reference = hist_reference.cumsum()/hist_reference.sum()

    lut = np.interp(cdf_image, cdf_reference, range(256))
    matched_image = cv2.LUT(image, lut.astype('uint8'))

    return matched_image

if __name__ == "__main__":
    ref = cv2.cvtColor(cv2.imread('2.bmp', cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(cv2.imread('1.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)

    ref_h, ref_s, ref_v =cv2.split(ref)
    img_h, img_s, img_v =cv2.split(img)

    matched_h = hist_matching(img_h, ref_h)

    matched_img = cv2.cvtColor(cv2.merge([matched_h, img_s, img_v]),cv2.COLOR_HSV2BGR)

    cv2.imwrite(f'./3.png', matched_img)