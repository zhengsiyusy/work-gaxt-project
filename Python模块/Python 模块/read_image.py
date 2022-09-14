# / -*-codeing = utf-8  -*-
# TIME : 2022/7/11 15:28
# File : read_image  读取视频每一帧图像，并将图像以特定名称保存
import cv2
import os
import time
cap = cv2.VideoCapture(0)
isOpened = cap.isOpened
print(isOpened)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps, width, height)

os.chdir(r"C:\Users\Dell\Desktop\healthy_code\image\1")

print(os.getcwd())
flag = True
i = 0
while True:
    if i == 300:
        break
    else:
        i = i + 1
    (flag, frame) = cap.read()   # 读取每一帧，flag表示是否读取成功，frame为图片内容。
    if flag:
        cv2.imshow("image", frame)
    fileName = "image" + str(i) + ".jpg"
    print(fileName)
    if flag:

        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        time.sleep(1)
    k = cv2.waitKey(1)
    if k == ord('q') or k == ord('Q'):
        break
os.chdir(r"C:\Users\Dell\Desktop\healthy_code\image")
print(os.getcwd())
print("end!")
cap.release()
cv2.destroyAllWindows()
