import os
import glob
import cv2
import numpy as np


extend_top = 5
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

class Capture:
    ScansPerFrame=208
    Blocks = 4
    BlockArray_frames = [40, 104, 168, 208]
    BlockArray_crops_top_pixels = [6, 20, 25, 20]
    BlockArray_crops_bottom_pixels = [56, 60, 55, 42]
    BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
    BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
    # BlockArray_overlap_top_pixels = [6, 20, 25, 20]
    # BlockArray_overlap_bottom_pixels = [56-1, 60-1, 55-1, 42-1]
    # BlockArray_overlap_top_pixels = [0, 0, 0, 0]
    # BlockArray_overlap_bottom_pixels = [0, 0, 0, 0]
    class Strip:
        Width=4608
        Height=96

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


def imgFusion(img1, img2, overlap_top, overlap_bottom, IF_BOTTOM):
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

    if not IF_BOTTOM:
        img_new = np.zeros((row1+row2 - overlap_top - overlap_bottom, col))
        print('img_new[:row1, :].shape, img1.shape')
        print(img_new[:row1, :].shape,img1.shape )
        img_new[:row1, :] = img1
        w = np.reshape(w, (overlap_top+overlap_bottom, 1))
        w_expand = np.tile(w, (1, col))
        img_new[row1 - overlap_top - overlap_bottom:row1, :] = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
        img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]
    else:
        img_new = np.zeros((row1+row2 - overlap_top - overlap_bottom, col))
        print('img_new[:row1, :].shape, img1.shape')
        print(img_new[:row1, :].shape,img1.shape )
        img_new[:row1, :] = img1
        w = np.reshape(w, (overlap_top+overlap_bottom, 1))
        w_expand = np.tile(w, (1, col))
        img_new[row1 - overlap_top - overlap_bottom:row1, :] = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
        img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]

    return img_new


def channelFusion(image_list, length):
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
            if i == 0:
                # Read the BMP strip
                img_up = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)

                img_up = img_up[up_crop_top:, :]
            elif i == length-1:
                img_up = img_up[:-up_crop_bottom, :]

                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img_down = img_down[down_crop_top:-down_crop_bottom, :]

                img_up = cv2.vconcat([img_up, img_down])
            else:
                img_up = img_up[:-up_crop_bottom, :]

                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img_down = img_down[down_crop_top:, :]

                img_up = cv2.vconcat([img_up, img_down])

        else:
            if i == 0:
                # Read the BMP strip
                img_up = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)

                img_up = img_up[up_crop_top:, :]
            elif i == length-1:
                img_up = img_up[:-(up_crop_bottom - overlap_bottom), :]
                print(-(up_crop_bottom - overlap_bottom))
                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                # to fix, for the last strip
                if i == len(image_list)-1:
                    img_down = img_down[down_crop_top - overlap_top:-down_crop_bottom, :]
                else:
                    img_down = img_down[down_crop_top - overlap_top:, :]
                print(down_crop_top - overlap_top, -down_crop_bottom)
                img_up = (img_up - img_up.min()) / img_up.ptp()
                img_down = (img_down - img_down.min()) / img_down.ptp()

                print('img_up.shape, img_down.shape')
                print(img_up.shape, img_down.shape)
                img_up = imgFusion(img_up, img_down, overlap_top=overlap_top, overlap_bottom=overlap_bottom, IF_BOTTOM=True)
                img_up = np.uint16(img_up * 65535)

            else:
                img_up = img_up[:-(up_crop_bottom - overlap_bottom), :]

                img_down = cv2.rotate(cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE), cv2.ROTATE_180)
                img_down = img_down[down_crop_top - overlap_top:, :]

                img_up = (img_up - img_up.min()) / img_up.ptp()
                img_down = (img_down - img_down.min()) / img_down.ptp()
                print('img_up.shape, img_down.shape')
                print(img_up.shape, img_down.shape)
                img_up = imgFusion(img_up, img_down, overlap_top=overlap_top, overlap_bottom=overlap_bottom, IF_BOTTOM=False)
                img_up = np.uint16(img_up * 65535)

    return img_up

path = r'..\Dataset\strip'
folder_list = glob.glob(f'{path}/*')

for folder in folder_list[2:3]:
    print(folder)
    image_list = glob.glob(f'{folder}/*.bmp')
    ScansPerFrame = int(len(image_list)/3)
    print(ScansPerFrame)
    R_list = image_list[:ScansPerFrame]
    G_list = image_list[ScansPerFrame:ScansPerFrame*2]
    B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

    length = len(R_list)
    # length = 2

    R_fusion = channelFusion(R_list, length)
    G_fusion = channelFusion(G_list, length)
    B_fusion = channelFusion(B_list, length)
    # print(R_fusion.shape, G_fusion.shape)
    # cv2.merge 实现图像通道的合并
    imgMerge = cv2.merge([B_fusion, G_fusion, R_fusion])

    # print(R_fusion.shape,G_fusion.shape,B_fusion.shape)

# cv2.imwrite(f'./blend-overlap-{overlap}.png', imgMerge)
save_file_name = f'{Capture.BlockArray_overlap_top_pixels[0]}-{Capture.BlockArray_overlap_top_pixels[1]}-{Capture.BlockArray_overlap_top_pixels[2]}-{Capture.BlockArray_overlap_top_pixels[3]}-{Capture.BlockArray_overlap_bottom_pixels[0]}-{Capture.BlockArray_overlap_bottom_pixels[0]}-{Capture.BlockArray_overlap_bottom_pixels[0]}-{Capture.BlockArray_overlap_bottom_pixels[0]}'
cv2.imwrite(f'./{save_file_name}.png', imgMerge)
# cv2.imshow('Stitched Image', img_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

