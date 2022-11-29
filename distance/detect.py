# -*- coding: utf-8 -*-
import numpy as np
import cv2
from hobot_dnn import pyeasy_dnn as dnn
from bputools.format_convert import imequalresize, bgr2nv12_opencv
import lib.pyyolotools as yolotools
from model_output_merge import modelprocess, draw_recognize_result
# import matlab_calibrateCamera as camera_configs
import video


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
    model_path = './models/yolov5_brick_320.bin'
    thre_confidence = 0.8
    thre_score = 0.25
    thre_nms = 0.45

    models = dnn.load(model_path)
    model_h, model_w = get_hw(models[0].inputs[0].properties)
    # print(model_h, model_w)

    # camera = cv2.VideoCapture(8)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    camera = video.VideoCamera()
    while True:
        imgOri = camera.start_record()
        # frame = imgOri.copy()
        camera.write_frame(imgOri)
        # ret, imgOri = camera.read()
        # if not ret:
        #     print("not found camera")
        #     break
        # h, w = imgOri.shape[:2]
        # print("shape", h, w)
        # 相机校准
        # img1_rectified = cv2.remap(imgOri, camera_configs.left_map1, camera_configs.left_map2,
        #                            cv2.INTER_LINEAR)
        # # img2_rectified = cv2.remap(right_frame, camera_configs.right_map1, camera_configs.right_map2,
        # #                            cv2.INTER_LINEAR)
        # # 裁剪图像
        # crop_img = img1_rectified[15:h - 15, 15:w - 15]
        # cv2.imwrite("save.jpg", crop_img)

        inputImage = format_yolov5(imgOri)
        img = imequalresize(inputImage, (model_w, model_h))
        nv12 = bgr2nv12_opencv(img)

        t1 = cv2.getTickCount()
        outputs = models[0].forward(nv12)

        new_outputs = modelprocess(outputs)
        image_width, image_height, _ = inputImage.shape
        fx, fy = image_width / model_w, image_height / model_h
        t3 = cv2.getTickCount()
        cv2.imshow("result", imgOri) 
        # try:
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
            camera.write_frame(imgOri)
            
            cv2.imwrite('res.jpg', imgOri)
            key = cv2.waitKey(1)
            if key == ord("q"):
                camera.stop_record() 
                break
        # except TypeError: 
        #     print("nothing found")
        # cv2.imshow("result", imgOri) 
        
    camera.release()
    cv2.destroyAllWindows()
