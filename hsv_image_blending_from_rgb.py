import os
import glob
import cv2
import numpy as np
from hist_matching import hist_matching
import time

# Note that overlap_top must not be larger than the number of valid pixels which is 16 in the current setting
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

# class Capture_General:
#     ScansPerFrame=208
#     Blocks = 4
#     BlockArray_frames = [40, 104, 168, 208]
#     BlockArray_crops_top_pixels = [6, 20, 25, 20]
#     BlockArray_crops_bottom_pixels = [56, 60, 55, 42]

# class Capture_General:
#     ScansPerFrame=208
#     Blocks = 4
#     BlockArray_frames = [40, 104, 168, 208]
#     BlockArray_crops_top_pixels = [12, 30, 25, 10]
#     BlockArray_crops_bottom_pixels = [50, 50, 55, 52]

# class Capture_General:
#     ScansPerFrame=298
#     Blocks = 4
#     BlockArray_frames = [85, 85+64, 85+64+64, 85+64+64+85]
#     BlockArray_crops_top_pixels = [10, 10, 15, 30]
#     BlockArray_crops_bottom_pixels = [70, 70, 65, 50]

class Capture_General:
    ScansPerFrame=298
    Blocks = 4
    BlockArray_frames = [85, 85+64, 85+64+64, 85+64+64+85]
    BlockArray_crops_top_pixels = [20, 30, 25, 20]
    BlockArray_crops_bottom_pixels = [60, 50, 55, 60]

# class Capture_General:
#     ScansPerFrame=298
#     Blocks = 4
#     BlockArray_frames = [85, 85+64, 85+64+64, 85+64+64+85]
#     BlockArray_crops_top_pixels = [25, 25, 30, 35]
#     BlockArray_crops_bottom_pixels = [55, 55, 50, 45]

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
    BlockArray_overlap_top_pixels = [0, 0, 0, 0]
    BlockArray_overlap_bottom_pixels = [0, 0, 0, 0]

class Capture_S:
    ScansPerFrame=Capture_General.ScansPerFrame
    Blocks = Capture_General.Blocks
    BlockArray_frames = Capture_General.BlockArray_frames
    BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
    # BlockArray_overlap_top_pixels = [x-extend_top for x in BlockArray_crops_top_pixels]
    # BlockArray_overlap_bottom_pixels = [x-extend_bottom for x in BlockArray_crops_bottom_pixels]
    # BlockArray_overlap_top_pixels = [6, 20, 25, 20]
    # BlockArray_overlap_bottom_pixels = [56-1, 60-1, 55-1, 42-1]
    BlockArray_overlap_top_pixels = [0, 0, 0, 0]
    BlockArray_overlap_bottom_pixels = [0, 0, 0, 0]

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
    # BlockArray_overlap_top_pixels = [16, 16, 16, 16]
    # BlockArray_overlap_bottom_pixels = [30, 30, 30, 30]

def calWeight(overlap_top, overlap_bottom):
    x = np.arange(-overlap_top, overlap_bottom)
    # y = 1 / (1 + np.exp(-k * x)) #exponenital
    y = x/(overlap_top+overlap_bottom) + overlap_top/(overlap_top+overlap_bottom) # linear
    return y


def imgBlend(img1, img2, overlap_top, overlap_bottom):

    w = calWeight(overlap_top, overlap_bottom)

    row1, col = img1.shape
    row2, col = img2.shape
    # print('img1.shape, img2.shape')
    # print(img1.shape, img2.shape)

    # method 1
    img_new = np.zeros((row1 + row2 - overlap_top - overlap_bottom, col))
    img_new[:row1 - overlap_top - overlap_bottom, :] = img1[:row1 - overlap_top - overlap_bottom, :]
    w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    w_expand = np.tile(w, (1, col))
    img_new[row1 - overlap_top - overlap_bottom:row1, :] = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
    img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]

    # # method 2
    # w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    # w_expand = np.tile(w, (1, col))
    #
    # t0 = time.time()
    # tmp = np.concatenate((img1[:row1 - overlap_top - overlap_bottom, :], (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :], ), axis=0)
    # t1 = time.time()
    # print(f"T1: {t1-t0}")
    # img_new = np.concatenate((tmp,img2[(overlap_top + overlap_bottom):, :]), axis=0)
    # t2 = time.time()
    # print(f"T2: {t2 - t1}")

    # # method 3
    # w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    # w_expand = np.tile(w, (1, col))
    #
    # t0 = time.time()
    # a = img1[:row1 - overlap_top - overlap_bottom, :]
    # b = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
    # c = img2[(overlap_top+overlap_bottom):, :]
    # print(b.shape, c.shape)
    # img_new = cv2.vconcat([b,c])
    # t1 = time.time()
    # print(f"T1: {t1-t0}")

    # # method 4
    # w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    # w_expand = np.tile(w, (1, col))
    # a = img1[:row1 - overlap_top - overlap_bottom, :]
    # b = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
    # c = img2[(overlap_top+overlap_bottom):, :]
    # print(a.shape, b.shape)
    # # print(a.dims, b.dims)
    # # print(a.cols, b.cols)
    # # print(a.type(),b.type())
    # img_new = np.vstack((np.vstack((a,b)), c))
    # print(img_new.shape)

    return img_new


def hsvChannelBlend(image_list, Capture_V):
    Capture = Capture_V
    for i in range(Capture.ScansPerFrame):
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
                img_up_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_up_R = img_up_R[:-up_crop_top, :]

                img_up_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                img_up_G = img_up_G[:-up_crop_top, :]

                img_up_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                img_up_B = img_up_B[:-up_crop_top, :]

            elif i == Capture.ScansPerFrame-1:
                img_up_R = img_up_R[up_crop_bottom:, :]
                img_down_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down_R = img_down_R[down_crop_bottom:-down_crop_top, :]
                img_up_R = cv2.vconcat([img_down_R, img_up_R])

                img_up_G = img_up_G[up_crop_bottom:, :]
                img_down_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                img_down_G = img_down_G[down_crop_bottom:-down_crop_top, :]
                img_up_G = cv2.vconcat([img_down_G, img_up_G])

                img_up_B = img_up_B[up_crop_bottom:, :]
                img_down_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                img_down_B = img_down_B[down_crop_bottom:-down_crop_top, :]
                img_up_B = cv2.vconcat([img_down_B, img_up_B])

                imgMerge_all = cv2.merge([img_up_B, img_up_G, img_up_R])

            else:
                img_up_R = img_up_R[up_crop_bottom:, :]
                img_down_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down_R = img_down_R[:-down_crop_top, :]
                img_up_R = cv2.vconcat([img_down_R, img_up_R])

                img_up_G = img_up_G[up_crop_bottom:, :]
                img_down_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                img_down_G = img_down_G[:-down_crop_top, :]
                img_up_G = cv2.vconcat([img_down_G, img_up_G])

                img_up_B = img_up_B[up_crop_bottom:, :]
                img_down_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                img_down_B = img_down_B[:-down_crop_top, :]
                img_up_B = cv2.vconcat([img_down_B, img_up_B])

        else:
            if i == 0:
                img_up_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_up_R = img_up_R[:-up_crop_top, :]

                img_up_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                img_up_G = img_up_G[:-up_crop_top, :]

                img_up_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                img_up_B = img_up_B[:-up_crop_top, :]

                imgMerge_up = cv2.merge([img_up_B, img_up_G, img_up_R])
                img_up_HSV = cv2.cvtColor(imgMerge_up, cv2.COLOR_BGR2HSV)

                img_up_H, img_up_S, img_up_V = cv2.split(img_up_HSV)

            elif i == Capture.ScansPerFrame-1:
                img_down_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                # img_down_R = img_down_R[:-(down_crop_top - overlap_top), :]

                img_down_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                # img_down_G = img_down_G[:-(down_crop_top - overlap_top), :]

                img_down_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                # img_down_B = img_down_B[:-(down_crop_top - overlap_top), :]

                imgMerge_down = cv2.merge([img_down_B, img_down_G, img_down_R])
                img_down_HSV = cv2.cvtColor(imgMerge_down, cv2.COLOR_BGR2HSV)

                img_down_H, img_down_S, img_down_V = cv2.split(img_down_HSV)

                img_up_V = img_up_V[up_crop_bottom - overlap_bottom:, :]
                img_down_V = img_down_V[:-(down_crop_top - overlap_top), :]

                # if normalization is removed, somehow the code throws an error. To be fixed.
                img_up_V = (img_up_V - img_up_V.min()) / img_up_V.ptp()
                img_down_V = (img_down_V - img_down_V.min()) / img_down_V.ptp()

                img_up_V = imgBlend(img_down_V, img_up_V, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                # if normalization is removed, somehow the code throws an error. To be fixed.
                img_up_V = np.uint8(img_up_V * 255)
                img_up_V = img_up_V[down_crop_bottom:, :]

                img_up_H = cv2.vconcat([img_down_H[down_crop_bottom:-down_crop_top, :], img_up_H[up_crop_bottom:,:]])
                img_up_S = cv2.vconcat([img_down_S[down_crop_bottom:-down_crop_top, :], img_up_S[up_crop_bottom:,:]])

                imgMerge_all = cv2.cvtColor(cv2.merge([img_up_H, img_up_S, img_up_V]), cv2.COLOR_HSV2BGR)

            else:
                img_down_R = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                # img_down_R = img_down_R[:-(down_crop_top - overlap_top), :]

                img_down_G = cv2.imread(image_list[i+Capture.ScansPerFrame], cv2.IMREAD_GRAYSCALE)
                # img_down_G = img_down_G[:-(down_crop_top - overlap_top), :]

                img_down_B = cv2.imread(image_list[i+Capture.ScansPerFrame*2], cv2.IMREAD_GRAYSCALE)
                # img_down_B = img_down_B[:-(down_crop_top - overlap_top), :]

                imgMerge_down = cv2.merge([img_down_B, img_down_G, img_down_R])
                img_down_HSV = cv2.cvtColor(imgMerge_down, cv2.COLOR_BGR2HSV)

                img_down_H, img_down_S, img_down_V = cv2.split(img_down_HSV)

                img_up_V = img_up_V[up_crop_bottom - overlap_bottom:, :]
                img_down_V = img_down_V[:-(down_crop_top - overlap_top), :]

                t0 = time.time()

                t1 = time.time()
                print(f"t1-t0:{t1 - t0}")

                img_up_V = imgBlend(img_down_V, img_up_V, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                t2 = time.time()
                print(f"t2-t1:{t2-t1}")

                t3 = time.time()
                print(f"t3-t2:{t3-t2}")

                img_up_H = cv2.vconcat([img_down_H[:-down_crop_top, :], img_up_H[up_crop_bottom:,:]])
                img_up_S = cv2.vconcat([img_down_S[:-down_crop_top, :], img_up_S[up_crop_bottom:,:]])


    return imgMerge_all

IMAGE_INDEX = 9

path = f'../Dataset/strip_rgb'
folder_list = glob.glob(f'{path}/*')

for folder in folder_list[IMAGE_INDEX:IMAGE_INDEX+1]:
    print(folder)
    image_list = glob.glob(f'{folder}/*.bmp')
    ScansPerFrame = Capture_General.ScansPerFrame

    imgMerge = hsvChannelBlend(image_list, Capture_V)

    save_file_name = f'{os.path.basename(folder)}-HV-{Capture_V.BlockArray_overlap_top_pixels[0]}-{Capture_V.BlockArray_overlap_top_pixels[1]}-{Capture_V.BlockArray_overlap_top_pixels[2]}-{Capture_V.BlockArray_overlap_top_pixels[3]}-{Capture_V.BlockArray_overlap_bottom_pixels[0]}-{Capture_V.BlockArray_overlap_bottom_pixels[1]}-{Capture_V.BlockArray_overlap_bottom_pixels[2]}-{Capture_V.BlockArray_overlap_bottom_pixels[3]}'

    cv2.imwrite(f'./{save_file_name}.png', cv2.flip(imgMerge, 0))

