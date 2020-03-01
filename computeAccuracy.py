import cv2
import numpy as np
import os

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc6 = []

imgNum = 1181

for i in range(1, imgNum):

    gt = cv2.imread(os.path.join('../datasets/work2_3/N2DGAN/groundtruth/lake_2881/', '{}.jpg'.format(i - 1)))
    img1 = cv2.imread(os.path.join('../datasets/enhance+bg/LLNet/GMM/foreground-lake2881/', '{}.png'.format(i)))
    img2 = cv2.imread(os.path.join('../datasets/enhance+bg/LLNet/subsense/foreground-lake2881', '{}.png'.format(i)))

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0

    un1 = 0
    un2 = 0
    un3 = 0
    un4 = 0
    un5 = 0
    un6 = 0

    # print(gt.shape)
    # print(img1.shape)
    # print(img2.shape)

    for p in range(256):
        for q in range(256):
            if gt[p, q, 0] > 125 and img1[p, q, 0] > 125:
                cnt1 = cnt1 + 1
            if gt[p, q, 0] > 125 and img2[p, q, 0] > 125:
                cnt2 = cnt2 + 1
            # if gt[p, q, 0] > 125 and img3[p, q, 0] > 125:
            #     cnt3 = cnt3 + 1
            # if gt[p, q, 0] > 125 and img4[p, q, 0] > 125:
            #     cnt4 = cnt4 + 1
            # if gt[p, q, 0] > 125 and img5[p, q, 0] > 125:
            #     cnt5 = cnt5 + 1
            # if gt[p, q, 0] > 125 and img6[p, q, 0] > 125:
            #     cnt6 = cnt6 + 1

            if gt[p, q, 0] > 125 or img1[p, q, 0] > 125:
                un1 = un1 + 1
            if gt[p, q, 0] > 125 or img2[p, q, 0] > 125:
                un2 = un2 + 1
            # if gt[p, q, 0] > 125 or img3[p, q, 0] > 125:
            #     un3 = un3 + 1
            # if gt[p, q, 0] > 125 or img4[p, q, 0] > 125:
            #     un4 = un4 + 1
            # if gt[p, q, 0] > 125 or img5[p, q, 0] > 125:
            #     un5 = un5 + 1
            # if gt[p, q, 0] > 125 or img6[p, q, 0] > 125:
            #     un6 = un6 + 1

    if not un1 == 0:
        acc1.append(cnt1 * 1.0 / un1)
    else:
        acc1.append(0.0)
    if not un2 == 0:
        acc2.append(cnt2 * 1.0 / un2)
    else:
        acc2.append(0.0)
    # if not un3 == 0:
    #     acc3.append(cnt3 * 1.0 / un3)
    # else:
    #     acc3.append(0.0)
    # if not un4 == 0:
    #     acc4.append(cnt4 * 1.0 / un4)
    # else:
    #     acc4.append(0.0)
    # if not un5 == 0:
    #     acc5.append(cnt5 * 1.0 / un5)
    # else:
    #     acc5.append(0.0)
    # if not un6 == 0:
    #     acc6.append(cnt6 * 1.0 / un6)
    # else:
    #     acc6.append(0.0)

    # if i > 50:
    #     break

print('mean={}, var={}, std={}'.format(np.mean(acc1), np.var(acc1), np.std(acc1)))
print('mean={}, var={}, std={}'.format(np.mean(acc2), np.var(acc2), np.std(acc2)))
# print('mean={}, var={}, std={}'.format(np.mean(acc3), np.var(acc3), np.std(acc3)))
# print('mean={}, var={}, std={}'.format(np.mean(acc4), np.var(acc4), np.std(acc4)))
# print('mean={}, var={}, std={}'.format(np.mean(acc5), np.var(acc5), np.std(acc5)))
# print('mean={}, var={}, std={}'.format(np.mean(acc6), np.var(acc6), np.std(acc6)))
