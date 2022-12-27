# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from hobot_dnn import pyeasy_dnn as dnn
from bputools.format_convert import imequalresize, bgr2nv12_opencv
import lib.pyyolotools as yolotools
from model_output_merge import modelprocess, draw_recognize_result

Code_Version = '0.0.2'
print('Code_Version =', Code_Version)

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


if __name__ == '__main__':
    path = './models/'
    path_list = os.listdir(path)
    model_load_status = False
    for filename in path_list:
        print("name:", filename)
        if os.path.splitext(filename)[1] == '.bin':
            model_path = path + filename
            models = dnn.load(model_path)
            model_h, model_w = get_hw(models[0].inputs[0].properties)
            model_load_status = True
        else:
            model_load_status = False
            print('models load failed')
    thre_confidence = 0.8
    thre_score = 0.25
    thre_nms = 0.45

    camera = cv2.VideoCapture(8)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # camera = video.VideoCamera()
    while model_load_status:

        ret, imgOri = camera.read()
        if not ret:
            print("not found camera")
            break
        h, w = imgOri.shape[:2]
        inputImage = format_yolov5(imgOri)
        img = imequalresize(inputImage, (model_w, model_h))
        nv12 = bgr2nv12_opencv(img)

        t1 = cv2.getTickCount()
        outputs = models[0].forward(nv12)

        new_outputs = modelprocess(outputs)
        image_width, image_height, _ = inputImage.shape
        fx, fy = image_width / model_w, image_height / model_h
        t3 = cv2.getTickCount()
        class_ids, confidences, boxes = yolotools.pypostprocess_yolov5(new_outputs, fx, fy,
                                                                       thre_confidence, thre_score, thre_nms)
        if class_ids is None and confidences is None and boxes is None:
            print("none")
        else:
            t4 = cv2.getTickCount()
            print('post consumption {0} ms'.format((t4 - t3) * 1000 / cv2.getTickFrequency()))

            # Calculate frames per second
            micro_seconds = (t4 - t1) / cv2.getTickFrequency()
            fps = int(1 / micro_seconds)
            print("Estimated frames per second : {0}".format(fps))

            imgOri = draw_recognize_result(imgOri, class_ids, confidences, boxes, fps)

            cv2.imshow("result", imgOri)
            cv2.imwrite('res.jpg', imgOri)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    camera.release()
    cv2.destroyAllWindows()
