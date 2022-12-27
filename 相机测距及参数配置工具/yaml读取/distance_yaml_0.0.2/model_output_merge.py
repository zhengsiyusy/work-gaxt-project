# -*- coding: utf-8 -*-
# file:model_output_merge.py
import os
import cv2
import numpy as np
import math
import yaml

# from bputools.yolo_postproc import modelout2predbbox, recover_boxes, nms, draw_bboxs
# from bputools.format_convert import imequalresize


classes_name_path = 'coco_classes.names'
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]


class WRYaml:
    """ yaml文件的读和写 """

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
                return matrix
        else:
            print("config file doesn't exist")


# 单应性矩阵求解
# 相机固定好的情况下采集的点对   Four corners of the book in destination image.
# pts_src = np.array([[43.501, 448.510, 1], [245.501, 27.500, 1], [416.500, 29.497, 1], [602.499, 447.488, 1]])
# # 虚拟坐标系下对应点对
# pts_dst = np.array([[-45, 42, 1], [-45, 286, 1], [45, 286, 1], [45, 42, 1]])
# # Calculate Homography h是变换矩阵 status是掩码
# h, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
# # print("h", h)


def yolov5_decoder(conv_output, num_anchors, num_classes, anchors, stride):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Five dimension output: [batch_size, num_anchors, output_size, output_size, 5 + num_classes]
    batch_size = conv_output.shape[0]
    output_size = conv_output.shape[-2]
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = np.tile(
        np.arange(output_size, dtype=np.int32)[:, np.newaxis],
        [1, output_size])
    x = np.tile(
        np.arange(output_size, dtype=np.int32)[np.newaxis, :],
        [output_size, 1])
    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                             axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, np.newaxis, :, :, :],
                      [batch_size, num_anchors, 1, 1, 1])
    xy_grid = xy_grid.astype(np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) * 2.0 - 0.5 + xy_grid) * stride
    pred_wh = (sigmoid(conv_raw_dwdh) *
               2.0) ** 2 * anchors[np.newaxis, :, np.newaxis, np.newaxis, :]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
    return decode_output


def modelprocess(model_output):
    # num of classes
    num_classes = 1
    anchors = np.array([
        10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
        373, 326
    ]).reshape((3, 3, 2))
    num_anchors = anchors.shape[0]
    strides = np.array([8, 16, 32])

    pred_sbbox = model_output[0].buffer.reshape([1, 40, 40, 3,
                                                 6]).transpose([0, 3, 1, 2, 4])
    pred_mbbox = model_output[1].buffer.reshape([1, 20, 20, 3,
                                                 6]).transpose([0, 3, 1, 2, 4])
    pred_lbbox = model_output[2].buffer.reshape([1, 10, 10, 3,
                                                 6]).transpose([0, 3, 1, 2, 4])

    pred_sbbox = yolov5_decoder(pred_sbbox, num_anchors, num_classes,
                                anchors[0], strides[0])
    pred_mbbox = yolov5_decoder(pred_mbbox, num_anchors, num_classes,
                                anchors[1], strides[1])
    pred_lbbox = yolov5_decoder(pred_lbbox, num_anchors, num_classes,
                                anchors[2], strides[2])
    pred_bbox_temp = np.concatenate([
        np.reshape(pred_sbbox, (-1, 5 + num_classes)),
        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
        np.reshape(pred_lbbox, (-1, 5 + num_classes))
    ], axis=0)

    pred_bbox = pred_bbox_temp.astype(np.float32)

    return pred_bbox


def draw_recognize_result(frame, class_ids, confidences, boxes, fps):
    h = read_matrix_cfg()
    unit_distance = 1.0  # 单位长度cm，栅格坐标下1个单位为1cm
    with open(classes_name_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
        tt1 = cv2.getTickCount()
        class_num = len(class_ids)
        if class_num == 1:
            for (classid, confidence, box) in zip(class_ids, confidences, boxes):

                color = colors[int(classid) % len(colors)]
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 0, 0))
                confidence_img = format(confidence, '.3f')

                # confidence
                cv2.putText(frame, str(confidence_img), (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 0, 255))
                # fps
                cv2.putText(frame, str(format(fps, '.1f')), (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 0, 255))
                print("class:", class_list[classid], ", ""confidence", confidence, ", ""position:", box)

                h1, w1 = frame.shape[:2]
                # distance calculate  middle point position
                x = box[0] + (box[2] / 2)
                y = box[1] + box[3]
                print("box, x, y", box, x, y)
                if (0 < x < w1) and (0 < y < h1):
                    dis_point = [x, y, 1]
                    if frame is not None:
                        # calculate position
                        dst_p_x = (dis_point[0] * h[0, 0] + dis_point[1] * h[0, 1] + dis_point[2] * h[0, 2]) / \
                                  (dis_point[0] * h[2, 0] + dis_point[1] * h[2, 1] + dis_point[2] * h[2, 2])
                        dst_p_y = (dis_point[0] * h[1, 0] + dis_point[1] * h[1, 1] + dis_point[2] * h[1, 2]) / \
                                  (dis_point[0] * h[2, 0] + dis_point[1] * h[2, 1] + dis_point[2] * h[2, 2])
                        distance = math.sqrt(dst_p_x ** 2 + dst_p_y ** 2)
                        text_x = "x:%.1fcm" % (dst_p_x / unit_distance)
                        text_y = "y:%.1fcm" % (dst_p_y / unit_distance)
                        text_z = "dis:%.1fcm" % (distance / unit_distance)

                        label = class_list[classid]
                        print('%s中心点坐标,(%d, %d)距离相机距离%0.1f cm' % (label, x, y, distance))

                        blk = np.zeros(frame.shape, np.uint8)
                        cv2.rectangle(blk, box, (255, 0, 0), -1)
                        frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
                        cv2.putText(frame, text_x, (box[0], box[1] + 40), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
                        cv2.putText(frame, text_y, (box[0], box[1] + 60), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
                        cv2.putText(frame, text_z, (box[0], box[1] + 80), cv2.FONT_ITALIC, 0.5, (0, 0, 0))

        tt6 = cv2.getTickCount()
        print('done! draw rect consumption {0} ms'.format((tt6 - tt1) * 1000 / cv2.getTickFrequency()))
    return frame
