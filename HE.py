import cv2
import numpy as np

# 对图像进行 均衡化
def equalize_transfrom(gray_img):
    return cv2.equalizeHist(gray_img)


b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
b_out = equalize_transfrom(b)
g_out = equalize_transfrom(g)
r_out = equalize_transfrom(r)
equa_out = np.stack((b_out, g_out, r_out), axis=-1)
# deaw_gray_hist(equa_out[:, :, 0])
cv2.imshow('equa_out', equa_out)
cv2.waitKey()
