import os
import glob
import cv2
import numpy as np


extend_top = 1
extend_bottom = 20

# class Capture:
#     ScansPerFrame=188
#     Blocks = 4
#     BlockArray_frames = [32, 94, 156, 188]
#     BlockArray_crops_top_pixels = [16, 40, 45, 20]
#     BlockArray_crops_bottom_pixels = [40, 40, 35, 36]
#     BlockArray_overlap_top_pixels = [x-1 for x in BlockArray_crops_top_pixels]
#     BlockArray_overlap_bottom_pixels = [x-1 for x in BlockArray_crops_bottom_pixels]
#     class Strip:
#         Width=4608
#         Height=96

# class Capture_General:
#     ScansPerFrame=208
#     Blocks = 4
#     BlockArray_frames = [40, 104, 168, 208]
#     BlockArray_crops_top_pixels = [6, 20, 25, 20]
#     BlockArray_crops_bottom_pixels = [56, 60, 55, 42]

class Capture_General:
    ScansPerFrame=208
    Blocks = 4
    BlockArray_frames = [40, 104, 168, 208]
    BlockArray_crops_top_pixels = [12, 30, 25, 10]
    BlockArray_crops_bottom_pixels = [50, 50, 55, 52]

# class Capture_General:
#     ScansPerFrame=298
#     Blocks = 4
#     BlockArray_frames = [85, 85+64, 85+64+64, 85+64+64+85]
#     BlockArray_crops_top_pixels = [10, 10, 15, 30]
#     BlockArray_crops_bottom_pixels = [70, 70, 65, 50]

# class Capture_General:
#     ScansPerFrame=298
#     Blocks = 4
#     BlockArray_frames = [85, 85+64, 85+64+64, 85+64+64+85]
#     BlockArray_crops_top_pixels = [20, 30, 25, 20]
#     BlockArray_crops_bottom_pixels = [60, 50, 55, 60]

class Capture_V:
    ScansPerFrame=Capture_General.ScansPerFrame
    Blocks = Capture_General.Blocks
    BlockArray_frames = Capture_General.BlockArray_frames
    BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
    BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
    BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
    # BlockArray_overlap_top_pixels = [6, 20, 25, 20]
    # BlockArray_overlap_bottom_pixels = [56-1, 60-1, 55-1, 42-1]
    # BlockArray_overlap_top_pixels = [0, 0, 0, 0]
    # BlockArray_overlap_bottom_pixels = [0, 0, 0, 0]

Capture_S = Capture_V

class Capture_H:
    ScansPerFrame=Capture_General.ScansPerFrame
    Blocks = Capture_General.Blocks
    BlockArray_frames = Capture_General.BlockArray_frames
    BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
    # BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
    # BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
    # BlockArray_overlap_top_pixels = [6, 20, 25, 20]
    # BlockArray_overlap_bottom_pixels = [56-1, 60-1, 55-1, 42-1]
    BlockArray_overlap_top_pixels = [1, 0, 0, 1]
    BlockArray_overlap_bottom_pixels = [1, 0, 0, 1]

class Capture_R:
    ScansPerFrame=Capture_General.ScansPerFrame
    Blocks = Capture_General.Blocks
    BlockArray_frames = Capture_General.BlockArray_frames
    BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
    BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
    BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
    # BlockArray_overlap_top_pixels = [6-1, 20-1, 25-1, 20-1]
    # BlockArray_overlap_bottom_pixels = [56-15, 60-20, 55-20, 42-20]
    # BlockArray_overlap_top_pixels = [0, 0, 0, 0]
    # BlockArray_overlap_bottom_pixels = [0, 0, 0, 0]

# class Capture_G:
#     ScansPerFrame=Capture_General.ScansPerFrame
#     Blocks = Capture_General.Blocks
#     BlockArray_frames = Capture_General.BlockArray_frames
#     BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
#     BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
#     # BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
#     # BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
#     # BlockArray_overlap_top_pixels = [6, 20, 25, 20]
#     # BlockArray_overlap_bottom_pixels = [56-1, 60-1, 55-1, 42-1]
#     BlockArray_overlap_top_pixels = [1, 0, 0, 1]
#     BlockArray_overlap_bottom_pixels = [1, 0, 0, 1]

Capture_G = Capture_R
Capture_B = Capture_R



def calWeight(overlap_top, overlap_bottom, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-overlap_top, overlap_bottom)
    # y = 1 / (1 + np.exp(-k * x)) #exponenital
    y = x/(overlap_top+overlap_bottom) + overlap_top/(overlap_top+overlap_bottom) # linear
    return y


def imgFusion(img1, img2, overlap_top, overlap_bottom):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    w = calWeight(overlap_top, overlap_bottom, 0.05)  # k=5 这里是超参

    # if left_right:  # 左右融合
    #     col, row = img2.shape
    #     img_new = np.zeros((row, 2 * col - overlap))
    #     img_new[:, :col] = img1
    #     w_expand = np.tile(w, (col, 1))  # 权重扩增
    #     img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col] + w_expand * img2[:, :overlap]
    #     img_new[:, col:] = img2[:, overlap:]
    # else:

    # 上下融合
    row1, col = img1.shape
    row2, col = img2.shape
    print('img1.shape, img2.shape')
    print(img1.shape, img2.shape)

    # hack for some special cases. Not sure if there is a bug now.
    if row1+row2 - overlap_top - overlap_bottom<row1:
        img_new = img1
    else:
        img_new = np.zeros((row1 + row2 - overlap_top - overlap_bottom, col))
        img_new[:row1, :] = img1
        w = np.reshape(w, (overlap_top+overlap_bottom, 1))
        w_expand = np.tile(w, (1, col))
        img_new[row1 - overlap_top - overlap_bottom:row1, :] = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
        img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]

    return img_new


def channelBlend(image_list, length, Capture):
    for i in range(length):
        print(i)
        if i < Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[0]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            overlap_top = Capture.BlockArray_overlap_top_pixels[0]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
        elif i == Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            overlap_top = Capture.BlockArray_overlap_top_pixels[1]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
        elif i < Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            overlap_top = Capture.BlockArray_overlap_top_pixels[1]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
        elif i == Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            overlap_top = Capture.BlockArray_overlap_top_pixels[2]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
        elif i < Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            overlap_top = Capture.BlockArray_overlap_top_pixels[2]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
        elif i == Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            overlap_top = Capture.BlockArray_overlap_top_pixels[3]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
        else:
            up_crop_top = Capture.BlockArray_crops_top_pixels[3]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            overlap_top = Capture.BlockArray_overlap_top_pixels[3]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[3]

        if (overlap_top+overlap_bottom) == 0:
            print("Only concat. No blending.")
            if i == 0:
                # Read the BMP strip
                img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_up = img_up[:-up_crop_top, :]
            elif i == length-1:
                img_up = img_up[up_crop_bottom:, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down = img_down[down_crop_bottom:-down_crop_top, :]

                img_up = cv2.vconcat([img_down, img_up])
            else:
                img_up = img_up[up_crop_bottom:, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down = img_down[:-down_crop_top, :]

                img_up = cv2.vconcat([img_down, img_up])

        else:
            if i == 0:
                # Read the BMP strip
                img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_up = img_up[:-up_crop_top, :]
            elif i == length-1:
                img_up = img_up[up_crop_bottom - overlap_bottom:, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                # to fix, for the last strip
                if i == len(image_list)-1:
                    img_down = img_down[down_crop_bottom:-(down_crop_top - overlap_top), :]
                else:
                    img_down = img_down[:-(down_crop_top - overlap_top), :]

                img_up = (img_up - img_up.min()) / img_up.ptp()
                img_down = (img_down - img_down.min()) / img_down.ptp()

                print('img_up.shape, img_down.shape')
                print(img_up.shape, img_down.shape)
                img_up = imgFusion(img_down, img_up, overlap_top=overlap_top, overlap_bottom=overlap_bottom)
                img_up = np.uint8(img_up * 255)

            else:
                img_up = img_up[up_crop_bottom - overlap_bottom:, :]
                # print(-(up_crop_bottom - overlap_bottom))

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down = img_down[:-(down_crop_top - overlap_top), :]
                # print(down_crop_top - overlap_top, -down_crop_bottom)
                img_up = (img_up - img_up.min()) / img_up.ptp()
                img_down = (img_down - img_down.min()) / img_down.ptp()
                # print('img_up.shape, img_down.shape')
                # print(img_up.shape, img_down.shape)
                img_up = imgFusion(img_down, img_up, overlap_top=overlap_top, overlap_bottom=overlap_bottom)
                img_up = np.uint8(img_up * 255)

    return img_up


def channelConcat(image_list, length, Capture):
    for i in range(length):
        print(i)
        if i < Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[0]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            overlap_top = Capture.BlockArray_overlap_top_pixels[0]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
        elif i == Capture.BlockArray_frames[0]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[0]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            overlap_top = Capture.BlockArray_overlap_top_pixels[1]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
        elif i < Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[1]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            overlap_top = Capture.BlockArray_overlap_top_pixels[1]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
        elif i == Capture.BlockArray_frames[1]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[1]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            overlap_top = Capture.BlockArray_overlap_top_pixels[2]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
        elif i < Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[2]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            overlap_top = Capture.BlockArray_overlap_top_pixels[2]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
        elif i == Capture.BlockArray_frames[2]:
            up_crop_top = Capture.BlockArray_crops_top_pixels[2]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            overlap_top = Capture.BlockArray_overlap_top_pixels[3]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
        else:
            up_crop_top = Capture.BlockArray_crops_top_pixels[3]
            up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            down_crop_top = Capture.BlockArray_crops_top_pixels[3]
            down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
            overlap_top = Capture.BlockArray_overlap_top_pixels[3]
            overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[3]

        print("Only concat. No blending.")
        if i == 0:
            # Read the BMP strip
            img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

            img_up = img_up[:-up_crop_top, :]
        elif i == length - 1:
            img_up = img_up[up_crop_bottom:, :]

            img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
            img_down = img_down[down_crop_bottom:-down_crop_top, :]

            img_up = cv2.vconcat([img_down, img_up])
        else:
            img_up = img_up[up_crop_bottom:, :]

            img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
            img_down = img_down[:-down_crop_top, :]

            img_up = cv2.vconcat([img_down, img_up])

    return img_up

BLEND_HSV = True
IMAGE_INDEX = 2
if not BLEND_HSV:
    path = f'../Dataset/strip_rgb'
    folder_list = glob.glob(f'{path}/*')

    for folder in folder_list[IMAGE_INDEX:IMAGE_INDEX+1]:
        print(folder)
        image_list = glob.glob(f'{folder}/*.bmp')
        ScansPerFrame = int(len(image_list)/3)
        print(ScansPerFrame)
        R_list = image_list[:ScansPerFrame]
        G_list = image_list[ScansPerFrame:ScansPerFrame*2]
        B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

        length = len(R_list)
        # length = 2

        R_blend = channelBlend(R_list, length, Capture_R)
        G_blend = channelBlend(G_list, length, Capture_G)
        B_blend = channelBlend(B_list, length, Capture_B)
        # print(R_fusion.shape, G_fusion.shape)
        # cv2.merge 实现图像通道的合并
        imgMerge = cv2.merge([B_blend, G_blend, R_blend])

    save_file_name = f'RGB-{Capture_R.BlockArray_overlap_top_pixels[0]}-{Capture_R.BlockArray_overlap_top_pixels[1]}-{Capture_R.BlockArray_overlap_top_pixels[2]}-{Capture_R.BlockArray_overlap_top_pixels[3]}-{Capture_R.BlockArray_overlap_bottom_pixels[0]}-{Capture_R.BlockArray_overlap_bottom_pixels[1]}-{Capture_R.BlockArray_overlap_bottom_pixels[2]}-{Capture_R.BlockArray_overlap_bottom_pixels[3]}'
    cv2.imwrite(f'./{save_file_name}.png', cv2.flip(imgMerge, 0))

else:
    path = f'../Dataset/strip_hsv'
    folder_list = glob.glob(f'{path}/*')

    for folder in folder_list[IMAGE_INDEX:IMAGE_INDEX+1]:
        print(folder)
        image_list = glob.glob(f'{folder}/*.png')
        ScansPerFrame = int(len(image_list)/3)
        print(ScansPerFrame)
        H_list = image_list[:ScansPerFrame]
        S_list = image_list[ScansPerFrame:ScansPerFrame*2]
        V_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

        length = len(V_list)
        # length = 2

        H_concat = channelConcat(H_list, length, Capture_H)
        S_concat = channelConcat(S_list, length, Capture_S)
        V_blend = channelBlend(V_list, length, Capture_V)

        imgMerge = cv2.cvtColor(cv2.merge([H_concat, S_concat, V_blend]),cv2.COLOR_HSV2BGR)

    save_file_name = f'HV-{Capture_V.BlockArray_overlap_top_pixels[0]}-{Capture_V.BlockArray_overlap_top_pixels[1]}-{Capture_V.BlockArray_overlap_top_pixels[2]}-{Capture_V.BlockArray_overlap_top_pixels[3]}-{Capture_V.BlockArray_overlap_bottom_pixels[0]}-{Capture_V.BlockArray_overlap_bottom_pixels[1]}-{Capture_V.BlockArray_overlap_bottom_pixels[2]}-{Capture_V.BlockArray_overlap_bottom_pixels[3]}'
    cv2.imwrite(f'./{save_file_name}.png', cv2.flip(imgMerge, 0))

