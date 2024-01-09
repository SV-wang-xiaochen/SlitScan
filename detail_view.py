import os
import glob
import cv2
import numpy as np
from hist_matching import hist_matching
import time

orig = cv2.imread('blend.png')
blur1 = cv2.imread('blend_blur_9.png')
blur2 = cv2.imread('blend_blur_15.png')

orig_crop = orig[0:1000,1800:2800]
blur1_crop = blur1[0:1000,1800:2800]
blur2_crop = blur2[0:1000,1800:2800]

img_save = cv2.hconcat(([orig_crop, blur1_crop, blur2_crop]))

cv2.imwrite('r1.png', img_save)

orig_crop = orig[1800:2800,1800:2800]
blur1_crop = blur1[1800:2800,1800:2800]
blur2_crop = blur2[1800:2800,1800:2800]

img_save = cv2.hconcat(([orig_crop, blur1_crop, blur2_crop]))

cv2.imwrite('r2.png', img_save)