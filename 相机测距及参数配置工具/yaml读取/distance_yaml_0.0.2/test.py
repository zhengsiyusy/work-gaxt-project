# / -*-codeing = utf-8  -*-
# TIME : 2022/12/27 8:24
# File : test
import os
import cv2
import numpy as np
import math
import yaml


class WRYaml:
    def read_yaml(self, filepath):
        """ 读取yaml里面里面的数据"""
        try:
            with open(filepath, encoding="utf-8") as f:
                return yaml.load(f, Loader=yaml.Loader)
        except Exception as error:
            print(f'读取yaml失败,错误如下:{error}')
            return False

    def write_yaml(self, data, filepath, mode='a'):
        """ 往yaml里面写入数据
            filepath:yaml文件名
            data:要写入的数据
            mode:写入方式: w,覆盖写入, a,追加写入
        """
        try:
            with open(filepath, mode, encoding="utf-8") as f:
                yaml.dump(data, f)
            return True
        except Exception as error:
            print(f'yaml文件写入失败,错误如下:\n{error}')
            return False


# 读取坐标信息
def read_matrix_cfg():
    wryaml = WRYaml()
    # 当前路径下
    path = './config/'
    path_list = os.listdir(path)
    for filename in path_list:
        # 分离文件名与后缀名，返回值以元组形式存放
        if os.path.splitext(filename)[1] == '.yaml':
            filepath = path + filename
            read_data = wryaml.read_yaml(filepath)
            if read_data:
                position = read_data['position']
                new_matrix = [position['c1_x'], position['c1_y'], position['c2_x'], position['c2_y'], position['c3_x'],
                              position['c3_y'], position['c4_x'], position['c4_y'], position['c5_x'], position['c5_y'],
                              position['c6_x'], position['c6_y'], position['c7_x'], position['c7_y'], position['c8_x'],
                              position['c8_y'], position['p1_u'], position['p1_v'], position['p2_u'], position['p2_v'],
                              position['p3_u'], position['p3_v'], position['p4_u'], position['p4_v'], position['p5_u'],
                              position['p5_v'], position['p6_u'], position['p6_v'], position['p7_u'], position['p7_v'],
                              position['p8_u'], position['p8_v']]

                # 计算点数(8组坐标点)
                point_num = int(len(new_matrix) / 4)
                print(point_num)
                # 获取列表前16个数
                src = np.array(new_matrix[:16])
                # 获取列表后16个数
                dst = np.array(new_matrix[16:32])
                # 字符串转换为float64
                src = src.astype(np.float64)
                dst = dst.astype(np.float64)
                for i in range(point_num):
                    # 在每一个坐标点末尾添加1
                    src = np.insert(src, ((2 * i) + 2 + i * 1), [1])
                    dst = np.insert(dst, ((2 * i) + 2 + i * 1), [1])
                # 转换为4行3列列表
                pts_src = src.reshape(point_num, 3)
                pts_dst = dst.reshape(point_num, 3)
                # 求解转换矩阵
                matrix, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
                print('matrix', matrix)
                return matrix
        else:
            print("config file doesn't exist")



read_matrix_cfg()
