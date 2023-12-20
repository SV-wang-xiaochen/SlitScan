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
    # BlockArray_overlap_top_pixels = [16, 16, 16, 16]
    # BlockArray_overlap_bottom_pixels = [16, 16, 16, 16]

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



def calWeight(overlap_top, overlap_bottom):
    x = np.arange(-overlap_top, overlap_bottom)
    # y = 1 / (1 + np.exp(-k * x)) #exponenital
    y = x/(overlap_top+overlap_bottom) + overlap_top/(overlap_top+overlap_bottom) # linear
    return y


def imgBlend(img1, img2, overlap_top, overlap_bottom):

    w = calWeight(overlap_top, overlap_bottom)

    row1, col = img1.shape
    row2, col = img2.shape

    img_new = np.zeros((row1 + row2 - overlap_top - overlap_bottom, col))
    img_new[:row1 - overlap_top - overlap_bottom, :] = img1[:row1 - overlap_top - overlap_bottom, :]
    w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    w_expand = np.tile(w, (1, col))
    img_new[row1 - overlap_top - overlap_bottom:row1, :] = (1 - w_expand) * img1[row1 - overlap_top - overlap_bottom:row1, :] + w_expand * img2[:(overlap_top+overlap_bottom), :]
    img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]

    return img_new


def rgbChannelBlend(image_list, start_length, end_length, Capture, concat_only):
    for i in range(start_length, end_length):
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

        if (overlap_top+overlap_bottom) == 0 or concat_only:
            print("Only concat. No blending.")
            if i == start_length:
                # Read the BMP strip
                img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_up = img_up[:-up_crop_top, :]
            elif i == end_length-1:
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
            if i == start_length:
                # Read the BMP strip
                img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_up = img_up[:-up_crop_top, :]
            elif i == end_length-1:
                img_up = img_up[up_crop_bottom - overlap_bottom:, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_down = img_down[:-(down_crop_top - overlap_top), :]

                img_up = imgBlend(img_down, img_up, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                img_up = img_up[down_crop_bottom:, :]

            else:
                img_up = img_up[up_crop_bottom - overlap_bottom:, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_down = img_down[:-(down_crop_top - overlap_top), :]


                t1 = time.time()

                img_up = imgBlend(img_down, img_up, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                t2 = time.time()
                print(f"t2-t1:{t2-t1}")

    return img_up


# def channelConcat(image_list, length, Capture):
#     for i in range(length):
#         print(i)
#         if i < Capture.BlockArray_frames[0]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[0]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[0]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[0]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
#         elif i == Capture.BlockArray_frames[0]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[0]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[0]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[1]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[1]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[0]
#         elif i < Capture.BlockArray_frames[1]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[1]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[1]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[1]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
#         elif i == Capture.BlockArray_frames[1]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[1]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[1]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[2]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[2]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[1]
#         elif i < Capture.BlockArray_frames[2]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[2]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[2]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[2]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
#         elif i == Capture.BlockArray_frames[2]:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[2]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[2]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[3]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[3]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[2]
#         else:
#             up_crop_top = Capture.BlockArray_crops_top_pixels[3]
#             up_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
#             down_crop_top = Capture.BlockArray_crops_top_pixels[3]
#             down_crop_bottom = Capture.BlockArray_crops_bottom_pixels[3]
#             overlap_top = Capture.BlockArray_overlap_top_pixels[3]
#             overlap_bottom = Capture.BlockArray_overlap_bottom_pixels[3]
#
#         print("Only concat. No blending.")
#         if i == 0:
#             # Read the BMP strip
#             img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
#
#             img_up = img_up[:-up_crop_top, :]
#         elif i == length - 1:
#             img_up = img_up[up_crop_bottom:, :]
#
#             img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
#             img_down = img_down[down_crop_bottom:-down_crop_top, :]
#
#             img_up = cv2.vconcat([img_down, img_up])
#         else:
#             img_up = img_up[up_crop_bottom:, :]
#
#             img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
#             img_down = img_down[:-down_crop_top, :]
#
#             img_up = cv2.vconcat([img_down, img_up])
#
#     return img_up

HIST_MATCHING = False
BLEND_HSV = False
IMAGE_INDEX = 10
if not BLEND_HSV:
    path = f'../Dataset/strip_rgb'
    folder_list = glob.glob(f'{path}/*')

    for folder in folder_list[IMAGE_INDEX:IMAGE_INDEX+4]:
        print(folder)
        image_list = glob.glob(f'{folder}/*.bmp')
        ScansPerFrame = int(len(image_list)/3)
        print(ScansPerFrame)
        R_list = image_list[:ScansPerFrame]
        G_list = image_list[ScansPerFrame:ScansPerFrame*2]
        B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

        # length = len(R_list)
        length = 200
        start_length = 0
        end_length = len(R_list)

        CONCAT_ONLY = False
        R_blend = rgbChannelBlend(R_list, start_length, end_length, Capture_R, CONCAT_ONLY)
        cv2.imwrite(f'./R.png', cv2.flip(R_blend, 0))
        G_blend = rgbChannelBlend(G_list, start_length, end_length, Capture_G, CONCAT_ONLY)
        cv2.imwrite(f'./G.png', cv2.flip(G_blend, 0))
        B_blend = rgbChannelBlend(B_list, start_length, end_length, Capture_B, CONCAT_ONLY)
        cv2.imwrite(f'./B.png', cv2.flip(B_blend, 0))
        # print(R_fusion.shape, G_fusion.shape)
        # cv2.merge 实现图像通道的合并
        imgMerge = cv2.merge([B_blend, G_blend, R_blend])

        save_file_name = f'{os.path.basename(folder)}-RGB-{Capture_R.BlockArray_overlap_top_pixels[0]}-{Capture_R.BlockArray_overlap_top_pixels[1]}-{Capture_R.BlockArray_overlap_top_pixels[2]}-{Capture_R.BlockArray_overlap_top_pixels[3]}-{Capture_R.BlockArray_overlap_bottom_pixels[0]}-{Capture_R.BlockArray_overlap_bottom_pixels[1]}-{Capture_R.BlockArray_overlap_bottom_pixels[2]}-{Capture_R.BlockArray_overlap_bottom_pixels[3]}'
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

        # length = len(H_list)
        length = 200
        start_length = 0
        end_length = len(H_list)

        CONCAT_ONLY = False
        H_blend = rgbChannelBlend(H_list, start_length, end_length, Capture_H, CONCAT_ONLY)
        cv2.imwrite(f'./H.png', cv2.flip(H_blend, 0))
        S_blend = rgbChannelBlend(S_list, start_length, end_length, Capture_S, CONCAT_ONLY)
        cv2.imwrite(f'./S.png', cv2.flip(S_blend, 0))
        V_blend = rgbChannelBlend(V_list, start_length, end_length, Capture_V, CONCAT_ONLY)
        cv2.imwrite(f'./V.png', cv2.flip(V_blend, 0))

        if HIST_MATCHING:
            CONCAT_ONLY = True
            ref_h = rgbChannelBlend(H_list, start_length, end_length, Capture_H, CONCAT_ONLY)

            matched_h = hist_matching(H_blend, ref_h)
            imgMerge = cv2.cvtColor(cv2.merge([matched_h, S_blend, V_blend]), cv2.COLOR_HSV2BGR)
            save_file_name = f'{os.path.basename(folder)}-HV-match-{Capture_V.BlockArray_overlap_top_pixels[0]}-{Capture_V.BlockArray_overlap_top_pixels[1]}-{Capture_V.BlockArray_overlap_top_pixels[2]}-{Capture_V.BlockArray_overlap_top_pixels[3]}-{Capture_V.BlockArray_overlap_bottom_pixels[0]}-{Capture_V.BlockArray_overlap_bottom_pixels[1]}-{Capture_V.BlockArray_overlap_bottom_pixels[2]}-{Capture_V.BlockArray_overlap_bottom_pixels[3]}'

        else:
            print(H_blend.shape, S_blend.shape, V_blend.shape)
            imgMerge = cv2.cvtColor(cv2.merge([H_blend, S_blend, V_blend]),cv2.COLOR_HSV2BGR)
            save_file_name = f'{os.path.basename(folder)}-HV-{Capture_V.BlockArray_overlap_top_pixels[0]}-{Capture_V.BlockArray_overlap_top_pixels[1]}-{Capture_V.BlockArray_overlap_top_pixels[2]}-{Capture_V.BlockArray_overlap_top_pixels[3]}-{Capture_V.BlockArray_overlap_bottom_pixels[0]}-{Capture_V.BlockArray_overlap_bottom_pixels[1]}-{Capture_V.BlockArray_overlap_bottom_pixels[2]}-{Capture_V.BlockArray_overlap_bottom_pixels[3]}'

        cv2.imwrite(f'./{save_file_name}.png', cv2.flip(imgMerge, 0))

