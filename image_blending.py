import os
import glob
import cv2
import numpy as np


class Capture:
    ScansPerFrame=188
    Blocks = 4
    BlockArray_frames = [32, 94, 156, 188]
    BlockArray_crops_top_pixels = [16, 40, 45, 20]
    BlockArray_crops_bottom_pixels = [40, 40, 35, 36]
    class Strip:
        Width=4608
        Height=96

def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    w = calWeight(overlap, 0.05)  # k=5 这里是超参

    if left_right:  # 左右融合
        col, row = img2.shape
        img_new = np.zeros((row, 2 * col - overlap))
        img_new[:, :col] = img1
        w_expand = np.tile(w, (col, 1))  # 权重扩增
        img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col] + w_expand * img2[:, :overlap]
        img_new[:, col:] = img2[:, overlap:]
    else:  # 上下融合
        row1, col = img1.shape
        row2, col = img2.shape

        img_new = np.zeros((row1+row2 - overlap, col))
        img_new[:row1, :] = img1
        w = np.reshape(w, (overlap, 1))
        w_expand = np.tile(w, (1, col))
        img_new[row1 - overlap:row1, :] = (1 - w_expand) * img1[row1 - overlap:row1, :] + w_expand * img2[:overlap, :]
        img_new[row1:, :] = img2[overlap:, :]

    return img_new


def channelFusion(image_list):
    for i in range(len(image_list)-1):
        print(i)
        if i <= Capture.BlockArray_frames[0]:
            crop_top = Capture.BlockArray_crops_top_pixels[3]
            crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
        elif i <= Capture.BlockArray_frames[1]:
            crop_top = Capture.BlockArray_crops_top_pixels[2]
            crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
        elif i <= Capture.BlockArray_frames[2]:
            crop_top = Capture.BlockArray_crops_top_pixels[1]
            crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
        else:
            crop_top = Capture.BlockArray_crops_top_pixels[0]
            crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]

        OVERLAP = 32

        if i == 0:
            # Read the BMP strip
            img1 = cv2.imread(image_list[i+1], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

            img1 = img1[:Capture.Strip.Height - crop_bottom+int(OVERLAP/2), :]
            img2 = img2[crop_top-int(OVERLAP/2):Capture.Strip.Height - crop_bottom+int(OVERLAP/2), :]

            img1 = (img1 - img1.min())/img1.ptp()
            img2 = (img2 - img2.min())/img2.ptp()
            img_new = imgFusion(img1,img2,overlap=OVERLAP,left_right=False)
            img_new = np.uint16(img_new*65535)

        else:
            img1 = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
            img1 = img1[:Capture.Strip.Height - crop_bottom + int(OVERLAP / 2), :]

            img_new = img_new[crop_top - int(OVERLAP / 2):, :]
            img1 = (img1 - img1.min()) / img1.ptp()
            img_new = (img_new - img_new.min())/img_new.ptp()

            img_new = imgFusion(img1, img_new, overlap=OVERLAP, left_right=False)
            img_new = np.uint16(img_new * 65535)

    return img_new

path = r'..\Dataset\strip'
folder_list = glob.glob(f'{path}/*')

for folder in folder_list[:1]:
    print(folder)
    image_list = glob.glob(f'{folder}/*.bmp')
    ScansPerFrame = int(len(image_list)/3)
    print(ScansPerFrame)
    R_list = image_list[:ScansPerFrame]
    G_list = image_list[ScansPerFrame:ScansPerFrame*2]
    B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

    R_fusion = channelFusion(R_list)
    G_fusion = channelFusion(G_list)
    B_fusion = channelFusion(B_list)
    # print(R_fusion.shape, G_fusion.shape)
    # cv2.merge 实现图像通道的合并
    imgMerge = cv2.merge([B_fusion, G_fusion, R_fusion])

    # print(R_fusion.shape,G_fusion.shape,B_fusion.shape)

cv2.imwrite(r'.\fusion.png',cv2.rotate(imgMerge, cv2.ROTATE_180))
# cv2.imshow('Stitched Image', img_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

