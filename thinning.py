import cv2
import cv2.ximgproc
import numpy as np
import glob, os

img_list = glob.glob(r"C:\Users\xiaochen.wang\Desktop\skeleton\*.png")

os.makedirs(R"C:\Users\xiaochen.wang\Desktop\skeleton\MORPH",exist_ok= True)
for img_path in img_list:

    img = cv2.imread(img_path, 0)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    base_name = os.path.basename(img_path).split('.')[0]
    save_path = f"C:/Users/xiaochen.wang/Desktop/skeleton/MORPH/{base_name}_MORPH.png"
    cv2.imwrite(save_path, skel)

os.makedirs(R"C:\Users\xiaochen.wang\Desktop\skeleton\ZHANGSUEN",exist_ok= True)
for img_path in img_list:

    img = cv2.imread(img_path, 0)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    skel = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_ZHANGSUEN)

    base_name = os.path.basename(img_path).split('.')[0]
    save_path = f"C:/Users/xiaochen.wang/Desktop/skeleton/ZHANGSUEN/{base_name}_ZHANGSUEN.png"
    cv2.imwrite(save_path, skel)

os.makedirs(R"C:\Users\xiaochen.wang\Desktop\skeleton\GUOHALL",exist_ok= True)
for img_path in img_list:

    img = cv2.imread(img_path, 0)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    skel = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_GUOHALL)

    base_name = os.path.basename(img_path).split('.')[0]
    save_path = f"C:/Users/xiaochen.wang/Desktop/skeleton/GUOHALL/{base_name}_GUOHALL.png"
    cv2.imwrite(save_path, skel)

