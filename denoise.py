import cv2
import numpy as np
import os

data_path = '/Users/mengyingying/Desktop/bjtu/TCSVT/N2DGAN/MSR/tree/'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

# with open('config.json', 'r') as f:
#     config = json.load(f)

for img_name in img_list:
    if not img_name.endswith('.jpg'):
        continue
    print(img_name)
    img = cv2.imread(os.path.join(data_path, img_name))
    # dst = cv2.GaussianBlur(img, (5, 5), 0)
    dst = cv2.GaussianBlur(img, (5, 5), 3)
    cv2.imwrite(os.path.join('./MSR/denoise-MSR/tree/', img_name), dst)








