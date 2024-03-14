import cv2
import numpy as np
import glob
import os
import json
import xlsxwriter

spacing = 23.4742
mark_thickness = 1

column_names = ['ring-1', 'ring-2', 'ring-3', 'ring-4', 'ring-5', 'ring-6', 'ring-7', 'ring-8', 'ring-9', 'ring-10']
row_names = []
def ringIndexToRetinaUm(n):
    a1_mm = 0.7072
    a3_mm = 0.000272
    a5_mm = 0.00000033
    R_mm = a1_mm * n + a3_mm * n**3 + a5_mm * n**5
    retina_um = 1000 * R_mm
    return retina_um

def ringMask(img_width, img_height, center_x, center_y, inner_diameter, outer_diameter):
    # Create a black image (all zeros) as the base for the mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Generate a grid of coordinates
    y_coords, x_coords = np.ogrid[:img_height, :img_width]

    # Calculate the distance of each pixel from the center
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    # Create the ring mask by setting pixels within the ring range to 255
    mask[(distances >= inner_diameter) & (distances <= outer_diameter)] = 1

    return mask

data_path = r"D:\Projects\Dataset\issue_9205\denoise"
result_path = r"D:\Projects\Dataset\issue_9205\results"
os.makedirs(result_path, exist_ok=True)

result_table = []

img_list = glob.glob(f'{data_path}/*.png')
for img_path in img_list:
    base_name = os.path.basename(img_path).split('.png')[0]
    row_names.append(base_name)
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    src_color = cv2.imread(img_path)
    src_remove_noise = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # print(src.shape[1])
    center_x = int(src.shape[1]/2)-2
    center_y = int(src.shape[0]/2)+1
    cv2.circle(src, (center_x, center_y), radius=0, color=(0, 0, 255), thickness=mark_thickness)

    mask_list = []
    result_row = []
    for i in range(1, 11):
        radius_x = int(ringIndexToRetinaUm(i)/spacing)
        # print(radius_x)
        inner_diameter = radius_x-(i+2)
        outer_diameter= int(radius_x+(i+2))
        cv2.circle(src_color, (center_x, center_y), radius=outer_diameter, color=(0, 0, 255), thickness=mark_thickness)
        cv2.circle(src_color, (center_x, center_y), radius=inner_diameter, color=(0, 255, 0), thickness=mark_thickness)
        mask = ringMask(src.shape[1], src.shape[0], center_x, center_y, inner_diameter, outer_diameter)
        # print(mask.shape, src.shape)

        src_masked = np.where(mask==1, src, 0)
        mask_list.append(mask)

        cv2.imwrite(f'{result_path}/{base_name}-ring{str(i)}.png', src_masked)

        # Calculate number and average of elements beyond 0
        count_beyond_0 = np.count_nonzero(src_masked > 0)
        average_beyond_0 = np.mean(src_masked[src_masked > 0])
        print(average_beyond_0)

        result_row.append(average_beyond_0)

    base_name = os.path.basename(img_path)
    cv2.imwrite(f'{result_path}/{base_name}', src_color)

    result_table.append(result_row)

    # Create a 2D NumPy array
    array_2d = np.array(result_table)

    # Create a new Excel file and add a worksheet
    workbook = xlsxwriter.Workbook(f'{result_path}/results.xlsx')
    worksheet = workbook.add_worksheet()

    # Write the 2D NumPy array to the worksheet
    for row_num, row_data in enumerate(array_2d):
        for col_num, value in enumerate(row_data):
            worksheet.write(row_num + 1, col_num + 1, value)

    # Write column and row indices

    for col_num, value in enumerate(column_names):
        worksheet.write(0, col_num + 1, value)

    for row_num, value in enumerate(row_names):
        worksheet.write(row_num + 1, 0, value)

    workbook.close()

    # # Uncomment to display the mask
    # cv2.imshow('Ring Mask', mask_list[0] * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




