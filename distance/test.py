# / -*-codeing = utf-8  -*-
# TIME : 2022/11/29 11:07
# File : test
import cv2
import numpy as np

new_matrix = []
with open('matrix_config.cfg', 'r') as cfg:
    matrix = cfg.readlines()
    for i in range(len(matrix)):
        temp = matrix[i].replace('\n', '')
        new_matrix.append(temp)
    src = np.array(new_matrix[:8])
    dst = np.array(new_matrix[8:16])
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    for i in range(4):
        src = np.insert(src, ((2*i) + 2 + i*1), [1])
        dst = np.insert(dst, ((2 * i) + 2 + i * 1), [1])
    pts_src = src.reshape(4, 3)
    pts_dst = dst.reshape(4, 3)
    h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
    print("h", h)
cfg.close()
