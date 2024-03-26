import os
import glob
import cv2
import numpy as np
from hist_matching import hist_matching
import time
import configparser

data_path = r'D:\Projects\Dataset\20240322\256\hank'
ini_path = f'{data_path}\IngresSettings_2024-03-22-19-15-19.937.ini'
img_path = f'{data_path}\Record-Capture-2024-03-22-19-15-18.788'

config = configparser.ConfigParser()
config.read(ini_path)

block_frames_0 = int(config.get('Hardware.Camera', 'Capture.BlockArray_0_frames'))
block_frames_1 = int(config.get('Hardware.Camera', 'Capture.BlockArray_1_frames'))
block_frames_2 = int(config.get('Hardware.Camera', 'Capture.BlockArray_2_frames'))
block_frames_3 = int(config.get('Hardware.Camera', 'Capture.BlockArray_3_frames'))
class Capture_General:
    ScansPerFrame = int(config.get('Hardware.Camera', 'Capture.ScansPerFrame'))
    Blocks = int(config.get('Hardware.Camera', 'Capture.Blocks'))
    BlockArray_frames = [block_frames_0, block_frames_0+block_frames_1, block_frames_0+block_frames_1+block_frames_2,
                         block_frames_0+block_frames_1+block_frames_2+block_frames_3]
    BlockArray_crops_top_pixels = [int(config.get('Hardware.Camera', 'Capture.BlockArray_0_crops_top_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_1_crops_top_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_2_crops_top_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_3_crops_top_pixels'))]
    BlockArray_crops_bottom_pixels = [int(config.get('Hardware.Camera', 'Capture.BlockArray_0_crops_bottom_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_1_crops_bottom_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_2_crops_bottom_pixels')),
                                   int(config.get('Hardware.Camera', 'Capture.BlockArray_3_crops_bottom_pixels'))]
class Capture_R:
    ScansPerFrame=Capture_General.ScansPerFrame
    Blocks = Capture_General.Blocks
    BlockArray_frames = Capture_General.BlockArray_frames
    BlockArray_crops_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_crops_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels
    BlockArray_overlap_top_pixels = Capture_General.BlockArray_crops_top_pixels
    BlockArray_overlap_bottom_pixels = Capture_General.BlockArray_crops_bottom_pixels

Capture_G = Capture_R
Capture_B = Capture_R

def calWeight(overlap_top, overlap_bottom):
    x = np.arange(-overlap_top, overlap_bottom)
    # y = 1 / (1 + np.exp(-k * x)) #exponenital
    y = x/(overlap_top+overlap_bottom) + overlap_top/(overlap_top+overlap_bottom) # linear
    return y

def maxAccumulatedIntenstyCurveIndex(overlap_region1, overlap_region2):
    rows = overlap_region1.shape[0]
    acc_array = np.zeros((rows, 4608))
    for i in range(rows):
        acc_region1 = np.sum(overlap_region1[:i],0)
        acc_region2 = np.sum(overlap_region2[i:],0)
        acc_array[i,:] = acc_region1 + acc_region2

    max_accumulated_intensity = np.argmax(acc_array, 0)

    return max_accumulated_intensity

def maxAccumulatedIntenstyIndex(column_region1, column_region2):
    rows = column_region1.shape[0]

    max_accumulated_intensity = 0
    max_accumulated_intensity_index = 0

    for i in range(0, rows):
        accumulated_intensity1 = np.sum(column_region1[:i])
        accumulated_intensity2 = np.sum(column_region2[i:])

        if (accumulated_intensity1+accumulated_intensity2) > max_accumulated_intensity:
            max_accumulated_intensity = accumulated_intensity1+accumulated_intensity2
            max_accumulated_intensity_index = i

    return max_accumulated_intensity_index

def imgBlend(img1, img2, label_matrix, i, overlap_top, overlap_bottom):

    w = calWeight(overlap_top, overlap_bottom)

    row1, col = img1.shape
    row2, col = img2.shape

    img_new = np.zeros((row1 + row2 - overlap_top - overlap_bottom, col))
    label_new = np.zeros((row1 + row2 - overlap_top - overlap_bottom, col))

    img_new[:row1 - overlap_top - overlap_bottom, :] = img1[:row1 - overlap_top - overlap_bottom, :]
    label_new[:row1 - overlap_top - overlap_bottom, :] = label_matrix[:row1 - overlap_top - overlap_bottom, :]
    w = np.reshape(w, (overlap_top+overlap_bottom, 1))
    w_expand = np.tile(w, (1, col))

    overlap_region1 = img1[row1 - overlap_top - overlap_bottom:row1, :]
    overlap_label1 = label_matrix[row1 - overlap_top - overlap_bottom:row1, :]
    overlap_region2 = img2[:(overlap_top+overlap_bottom), :]
    cv2.imwrite(f'./curve/curve-{i}-up.png', overlap_region1)
    cv2.imwrite(f'./curve/curve-{i}-down.png', overlap_region2)
    # curve_list = maxAccumulatedIntenstyCurveIndex(overlap_region1, overlap_region2)
    # print(curve_list)
    #
    # curve_img = np.zeros((overlap_top+overlap_bottom, 4608))
    # curve_img_mask = np.zeros((overlap_top + overlap_bottom, 4608))
    # for k, index in enumerate(curve_list):
    #     curve_img[:index, k] = 255
    #     curve_img_mask[:index, k] = 1
    # cv2.imwrite(f'./curve/curve-{i}.png', curve_img)
    # #
    # # overlap_region = (1-w)*overlap_region1 + w*overlap_region2
    # # cv2.imwrite(f'./curve/curve-{i}-up-down-merged.png', overlap_region)
    #
    # img_new[row1 - overlap_top - overlap_bottom:row1, :] = curve_img_mask * overlap_region1 + (1-curve_img_mask) * overlap_region2
    # cv2.imwrite(f'./curve/curve-{i}-up-down-merged-new.png', img_new[row1 - overlap_top - overlap_bottom:row1, :])

    img_new[row1 - overlap_top - overlap_bottom:row1, :] = np.maximum(overlap_region1, overlap_region2)
    label_new[row1 - overlap_top - overlap_bottom:row1, :] = np.where(overlap_region1 >= overlap_region2, overlap_label1, i)

    cv2.imwrite(f'./curve/curve-{i}-up-down-merged-max.png', img_new[row1 - overlap_top - overlap_bottom:row1, :])

    img_new[row1:, :] = img2[(overlap_top+overlap_bottom):, :]
    label_new[row1:, :] = i

    return img_new, label_new


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

            elif i == end_length-1:
                img_up = img_up[:-up_crop_bottom, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down = img_down[down_crop_top:-down_crop_bottom, :]

                img_up = cv2.vconcat([img_up, img_down])
            else:
                img_up = img_up[:-up_crop_bottom, :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                img_down = img_down[down_crop_top:, :]

                img_up = cv2.vconcat([img_up, img_down])

        else:
            if i == start_length:
                # Read the BMP strip
                img_up = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
                label_matrix = np.zeros(img_up.shape)
            elif i == end_length-1:
                img_up = img_up[:img_up.shape[0]-(up_crop_bottom - overlap_bottom), :]
                label_matrix = label_matrix[:img_up.shape[0] - (up_crop_bottom - overlap_bottom), :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_down = img_down[down_crop_top - overlap_top:, :]

                img_up, label_matrix = imgBlend(img_up, img_down, label_matrix, i, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                img_up = img_up[:-down_crop_bottom, :]

            else:
                img_up = img_up[:img_up.shape[0]-(up_crop_bottom - overlap_bottom), :]
                label_matrix = label_matrix[:img_up.shape[0]-(up_crop_bottom - overlap_bottom), :]

                img_down = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)

                img_down = img_down[down_crop_top - overlap_top:, :]


                t1 = time.time()

                img_up, label_matrix = imgBlend(img_up, img_down, label_matrix, i, overlap_top=overlap_top, overlap_bottom=overlap_bottom)

                t2 = time.time()
                print(f"t2-t1:{t2-t1}")

    return img_up[Capture.BlockArray_crops_top_pixels[0]:, :], label_matrix[Capture.BlockArray_crops_top_pixels[0]:, :]

path = r'D:\Projects\Dataset\20240322\256\hank\Record-Capture-2024-03-22-19-15-18.788'
image_list = glob.glob(f'{path}/*.bmp')


ScansPerFrame = int(len(image_list)/3)
print(ScansPerFrame)
R_list = image_list[:ScansPerFrame]
G_list = image_list[ScansPerFrame:ScansPerFrame*2]
B_list = image_list[2*ScansPerFrame:ScansPerFrame*3]

start_length = 0
# end_length = 4
end_length = len(R_list)

CONCAT_ONLY = False
# R_blend = rgbChannelBlend(R_list, start_length, end_length, Capture_R, CONCAT_ONLY)
# cv2.imwrite(f'./R.png', cv2.flip(R_blend, 0))
G_blend, label_map = rgbChannelBlend(G_list, start_length, end_length, Capture_G, CONCAT_ONLY)
cv2.imwrite(f'./G.png', G_blend)
# B_blend = rgbChannelBlend(B_list, start_length, end_length, Capture_B, CONCAT_ONLY)
# cv2.imwrite(f'./B.png', B_blend)
# print(R_fusion.shape, G_fusion.shape)
# cv2.merge 实现图像通道的合并
# imgMerge = cv2.merge([B_blend, G_blend, R_blend])

save_file_name = f'RGB-{Capture_R.BlockArray_overlap_top_pixels[0]}-{Capture_R.BlockArray_overlap_top_pixels[1]}-{Capture_R.BlockArray_overlap_top_pixels[2]}-{Capture_R.BlockArray_overlap_top_pixels[3]}-{Capture_R.BlockArray_overlap_bottom_pixels[0]}-{Capture_R.BlockArray_overlap_bottom_pixels[1]}-{Capture_R.BlockArray_overlap_bottom_pixels[2]}-{Capture_R.BlockArray_overlap_bottom_pixels[3]}'
cv2.imwrite(f'./{save_file_name}.png', G_blend)
print(f'./{save_file_name}.png')
cv2.imwrite(f'./{save_file_name}-label.png', label_map)
