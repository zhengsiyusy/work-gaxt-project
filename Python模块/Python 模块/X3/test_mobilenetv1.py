from hobot_dnn import pyeasy_dnn
import numpy as np
import cv2


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


if __name__ == '__main__':
    # test classification result
    # models = pyeasy_dnn.load('mobilenetv1_224x224_nv12.bin')
    models = pyeasy_dnn.load('yolov5_672x672_nv12.bin')
    h, w = get_hw(models[0].inputs[0].properties)
    # img_file = cv2.imread('./zebra_cls.jpg')
    img_file = cv2.imread('./1.jpg')
    des_dim = (w, h)
    resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)
    outputs = models[0].forward(nv12_data)
    print("=" * 10, "Classification result", "=" * 10)
    # assert np.argmax(outputs[0].buffer) == 340
    print("cls id:", np.argmax(outputs[0].buffer))

    # test input and output properties
    print("=" * 10, "inputs[0] properties", "=" * 10)
    print_properties(models[0].inputs[0].properties)
    print("inputs[0] name is:", models[0].inputs[0].name)

    print("=" * 10, "outputs[0] properties", "=" * 10)
    print_properties(models[0].outputs[0].properties)
    print("outputs[0] name is:", models[0].outputs[0].name)

    # infer with random data according to input properties
    print("=" * 10, "Infer with assigned numpy data", "=" * 10)
    # as for nv12 type, input shape should be (h*1.5, w)
    tensor = np.ones((336,224), dtype='uint8')
    outputs = models[0].forward(tensor)

    print("=" * 10, "Get output[0] numpy data", "=" * 10)
    print("output[0] buffer numpy info: ")
    print("shape: ", outputs[0].buffer.shape)
    print("dtype: ", outputs[0].buffer.dtype)
