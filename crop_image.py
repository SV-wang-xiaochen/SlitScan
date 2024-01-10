import cv2, glob, os

path = r'C:\Users\xiaochen.wang\Desktop\live_view'
file_list = glob.glob(f'{path}\*.bmp')

for item in file_list:
    print(item)
    img = cv2.imread(item, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_channel = img_HSV[:, :, 2]

    # for crop in [1200, 1400, 1600]:
    crop = 1200
    basename = os.path.basename(item)
    cv2.imwrite(f'{path}/{basename}_2400.png', v_channel[int(height/2)-crop:int(height/2)+crop, int(width/2)-crop:int(width/2)+crop])


    # cv2.imwrite(f'./full.png', v_channel)