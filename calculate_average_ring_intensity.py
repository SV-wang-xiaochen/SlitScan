import cv2
import numpy as np
import glob
import os
import json

spacing = 23.4742
mark_thickness = 1
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

data_path = r"D:\Projects\Dataset\temp_issue"
result_path = r"D:\Projects\Dataset\temp_issue\results"
os.makedirs(result_path, exist_ok=True)

img_path = f"{data_path}/2.5mm VG-VC0296_2024-03-12_14-22-02_OD_Angio 15x12 640x512 R4.VG.png"
# img_path = f"{data_path}/1.7mm VG-VC0296_2024-03-12_14-10-09_OD_Angio 15x12 640x512 R4.VG.png"
src = cv2.imread(img_path)
print(src.shape[1])
center_x = int(src.shape[1]/2)-1
center_y = int(src.shape[0]/2)+1
cv2.circle(src, (center_x, center_y), radius=0, color=(0, 0, 255), thickness=mark_thickness)

mask_list = []
for i in range(1, 9):
    radius_x = int(ringIndexToRetinaUm(i)/spacing)
    print(radius_x)
    inner_diameter = radius_x-1
    outer_diameter= int(radius_x+(i+1)*2)
    cv2.circle(src, (center_x, center_y), radius=outer_diameter, color=(0, 0, 255), thickness=mark_thickness)
    cv2.circle(src, (center_x, center_y), radius=inner_diameter, color=(0, 0, 255), thickness=mark_thickness)
    mask = ringMask(src.shape[1], src.shape[0], center_x, center_y, inner_diameter, outer_diameter)
    mask_list.append(mask)

base_name = os.path.basename(img_path)
cv2.imwrite(f'{result_path}/{base_name}', src)

# # Uncomment to display the mask
# cv2.imshow('Ring Mask', mask_list[0] * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




