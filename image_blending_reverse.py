import os
import glob
import cv2
import numpy as np


class Capture:
    ScansPerFrame=188
    Blocks = 4
    BlockArray_frames = [32-1, 94-1, 156-1, 188-1]
    BlockArray_crops_top_pixels = [16, 40, 45, 20]
    BlockArray_crops_bottom_pixels = [40, 40, 35, 36]
    # BlockArray_crops_top_pixels = [0, 0, 0, 0]
    # BlockArray_crops_bottom_pixels = [0, 0, 0, 0]
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


def channelFusion(image_list, length, overlap):
    for i in range(length-1):
        print(i)
        if i < Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[0]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
        elif i == Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
        elif i < Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
        elif i == Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
        elif i < Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
        elif i == Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
        else:
            up_crop_top = Capture.BlockArray_crops_top_pixels[3]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]

        if overlap == 0:
            if i == 0:
                # Read the BMP strip
                img1 = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img2 = cv2.rotate(cv2.imread(image_list[i+1], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)

                img1 = img1[up_crop_top:Capture.Strip.Height - up_crop_bottom, :]
                img2 = img2[down_crop_top:, :]

                img_up = cv2.vconcat([img1, img2])
            else:
                img_up = img_up[:-up_crop_bottom, :]

                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img_down = img_down[down_crop_top:, :]

                img_up = cv2.vconcat([img_up, img_down])
        else:
            if i == 0:
                # Read the BMP strip
                img1 = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img2 = cv2.rotate(cv2.imread(image_list[i+1], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)

                img1 = img1[up_crop_top:Capture.Strip.Height - up_crop_bottom + int(overlap/2), :]
                img2 = img2[down_crop_top - int(overlap / 2):, :]

                img1 = (img1 - img1.min())/img1.ptp()
                img2 = (img2 - img2.min())/img2.ptp()
                img_up = imgFusion(img1,img2,overlap=overlap,left_right=False)
                img_up = np.uint16(img_up*65535)

            else:
                img_up = img_up[:-(up_crop_bottom - int(overlap / 2)), :]

                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img_down = img_down[down_crop_top - int(overlap / 2):, :]

                img_down = (img_down - img_down.min()) / img_down.ptp()
                img_up = (img_up - img_up.min())/img_up.ptp()

                img_up = imgFusion(img_up, img_down, overlap=overlap, left_right=False)
                img_up = np.uint16(img_up * 65535)

    return img_up

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

    overlap = 0
    length = len(R_list)

    R_fusion = channelFusion(R_list, length, overlap)
    G_fusion = channelFusion(G_list, length, overlap)
    B_fusion = channelFusion(B_list, length, overlap)
    # print(R_fusion.shape, G_fusion.shape)
    # cv2.merge 实现图像通道的合并
    imgMerge = cv2.merge([B_fusion, G_fusion, R_fusion])

    # print(R_fusion.shape,G_fusion.shape,B_fusion.shape)

cv2.imwrite(r'.\fusion.png', imgMerge)
# cv2.imshow('Stitched Image', img_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

