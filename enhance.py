import sys
import os
import numpy as np
import cv2
import json

import retinex

# 对图像进行 均衡化
def equalize_transfrom(gray_img):
    return cv2.equalizeHist(gray_img)


data_path = '/Users/mengyingying/Desktop/bjtu/TCSVT/datasets/work2_3/N2DGAN/tree/object2_256_256/'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

# with open('config.json', 'r') as f:
#     config = json.load(f)

for img_name in img_list:
    if not img_name.endswith('jpg'):
        continue
    print(img_name)
    img = cv2.imread(os.path.join(data_path, img_name))
    # print(os.path.join('./He/lake2881', img_name))
    # img_msr = retinex.multiScaleRetinex(img, [15, 80, 200])

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    b_out = equalize_transfrom(b)
    g_out = equalize_transfrom(g)
    r_out = equalize_transfrom(r)
    img_he = np.stack((b_out, g_out, r_out), axis=-1)
    cv2.imwrite(os.path.join('./HE/tree/', img_name), img_he)

    # print('msrcr processing......')
    img_msrcr = retinex.MSRCR(
        img,
        [15, 80, 200],
        5.0,
        25.0,
        125.0,
        46.0,
        0.01,
        0.99
    )
    # cv2.imshow('MSRCR retinex', img_msrcr)
    cv2.imwrite(os.path.join('./MSR/tree/', img_name), img_msrcr)
    continue


# ------------------------------------------------------






    print('amsrcr processing......')
    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )
    cv2.imshow('autoMSRCR retinex', img_amsrcr)
    cv2.imwrite('AutomatedMSRCR_retinex.tif', img_amsrcr)

    print('msrcp processing......')
    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    shape = img.shape
    cv2.imshow('Image', img)

    cv2.imshow('MSRCP', img_msrcp)
    cv2.imwrite('MSRCP.tif', img_msrcp)
    cv2.waitKey()
    break
