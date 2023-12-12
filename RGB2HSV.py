import os
import glob
import cv2
import numpy as np

path = f'../Dataset/strip_rgb'
folder_list = glob.glob(f'{path}/*')

for folder in folder_list[0:4]:
    save_folder = f'../Dataset/strip_hsv/{os.path.basename(folder)}'
    print(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    image_list = glob.glob(f'{folder}/*.bmp')
    ScansPerFrame = int(len(image_list)/3)
    print(ScansPerFrame)
    R_list = image_list[:ScansPerFrame]
    G_list = image_list[ScansPerFrame:ScansPerFrame*2]
    B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

    length = len(R_list)

    for i in range(len(R_list)):
        r = cv2.imread(R_list[i], cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(G_list[i], cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(B_list[i], cv2.IMREAD_GRAYSCALE)
        imgMerge = cv2.merge([b, g, r])
        img_HSV = cv2.cvtColor(imgMerge, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(img_HSV)

        base_name = os.path.basename(R_list[i]).split('.')[0][:-1]
        save_file_name_H = f'{save_folder}/{base_name}H.png'
        print(save_file_name_H)
        cv2.imwrite(save_file_name_H, h)

        base_name = os.path.basename(G_list[i]).split('.')[0][:-1]
        save_file_name_S = f'{save_folder}/{base_name}S.png'
        print(save_file_name_S)
        cv2.imwrite(save_file_name_S, s)

        base_name = os.path.basename(B_list[i]).split('.')[0][:-1]
        save_file_name_V = f'{save_folder}/{base_name}V.png'
        print(save_file_name_V)
        cv2.imwrite(save_file_name_V, v)

