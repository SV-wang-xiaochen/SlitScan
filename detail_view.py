import os
import glob
import cv2
import numpy as np
from hist_matching import hist_matching
import time

i1= cv2.imread('i1.png')
i2= cv2.imread('i2.png')

cv2.imwrite('i1-1.png', i1[0:1000,1800:2800])
cv2.imwrite('i2-1.png', i2[0:1000,1800:2800])
