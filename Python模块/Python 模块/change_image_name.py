# / -*-codeing = utf-8  -*-
# TIME : 2022/7/21 8:58
# File : TEST 将rootdir路径下的照片重新命名
import os
import os.path
import numpy as np
# 末尾斜杠不要丢，你的文件存放位置
# rootdir = r"D:\YOLOV5\VOC2007\JPEGImages/"
rootdir = r"C:\Users\Dell\Desktop\frame\22/"
files = os.listdir(rootdir)
# 可更改起始序号
b = 0
for name in files:
    a = os.path.splitext(name)
    print(a[0])
    newname = str(b)+'.jpg'
    b = b + 1
    os.rename(rootdir+name, rootdir+newname)