# -*- coding: utf-8 -*-
# file:model_output_merge.py
# 
import cv2
import numpy as np
import math
# from bputools.yolo_postproc import modelout2predbbox, recover_boxes, nms, draw_bboxs
# from bputools.format_convert import imequalresize


classes_name_path = 'coco_classes.names'
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

pts_src = np.array([[43.501, 448.510, 1], [245.501, 27.500, 1], [416.500, 29.497, 1], [602.499, 447.488, 1]])
# 虚拟坐标系下对应点对
pts_dst = np.array([[-45, 42, 1], [-45, 286, 1], [45, 286, 1], [45, 42, 1]])

# 读取转换矩阵
# def read_matrix_cfg():
#     new_matrix = []
#     with open('matrix_config.cfg', 'r') as cfg:
#         cfg_data = cfg.readlines()
#         for i in range(len(cfg_data)):
#             temp = cfg_data[i].replace('\n', '')
#             new_matrix.append(temp)
#         np_arr = np.array(new_matrix)
#         matrix_float = np_arr.astype(np.float64)
#         matrix = matrix_float.reshape(3, 3)
#         print(matrix)
#     cfg.close()
#     return matrix


# 读取坐标信息
def read_matrix_cfg():
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
            src = np.insert(src, ((2 * i) + 2 + i * 1), [1])
            dst = np.insert(dst, ((2 * i) + 2 + i * 1), [1])
        pts_src = src.reshape(4, 3)
        pts_dst = dst.reshape(4, 3)
        matrix, status = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
        print("matrix", matrix)
    cfg.close()
    return matrix


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
    # 读取配置文件中的单应性变换矩阵
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
