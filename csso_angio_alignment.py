import cv2
import numpy as np
import glob
import os
import json

data_path = r"D:\Projects\Dataset\AngioAnnotation\labeled"
result_path = f'{data_path}/results'
os.makedirs(result_path, exist_ok=True)

img_list_all = glob.glob(f'{data_path}/*.png')
img_list_csso = []
img_list_angio = []

for i, img_path in enumerate(img_list_all):
    if i%2 == 0:
        img_list_csso.append(img_path)
    else:
        img_list_angio.append(img_path)

json_list_all = glob.glob(f'{data_path}/*.json')
json_list_csso = []
json_list_angio = []

for i, json_path in enumerate(json_list_all):
    if i%2 == 0:
        json_list_csso.append(json_path)
    else:
        json_list_angio.append(json_path)

for i in range(len(img_list_csso)):
    json_path_csso = json_list_csso[i]
    json_path_angio = json_list_angio[i]

    with open(json_path_csso) as json_file:
        data = json.load(json_file)

    srcTri = []
    for shape in data['shapes']:
        srcTri.append(shape['points'][0])
    srcTri = np.array(srcTri).astype(np.float32)

    with open(json_path_angio) as json_file:
        data = json.load(json_file)
    dstTri = []
    for shape in data['shapes']:
        dstTri.append(shape['points'][0])
    dstTri = np.array(dstTri).astype(np.float32)

    src = cv2.imread(img_list_csso[i])
    dstRef = cv2.imread(img_list_angio[i])

    warp_mat = cv2.getAffineTransform(srcTri, dstTri)

    dstWarp = cv2.warpAffine(src, warp_mat, (dstRef.shape[1], dstRef.shape[0]))

    # Create an output image with the same size as the input images
    output = np.zeros_like(dstRef)

    # Assign different colors for each image
    # output[:, :, 0] = dstWarp[:, :, 0]
    output[:, :, 1] = dstWarp[:, :, 1]
    output[:, :, 2] = dstRef[:, :, 2]

    base_name = os.path.basename(img_list_csso[i])
    cv2.imwrite(f'{result_path}/{base_name}', output)
