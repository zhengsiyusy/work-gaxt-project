# / -*-codeing = utf-8  -*-
# TIME : 2022/9/13 11:44
# File : test
import os
from zipfile import ZipFile
unzip_path = "./unzip"
if not os.path.exists(unzip_path):
  os.mkdir(unzip_path)
with ZipFile("image/1662367068942图片数据.xlsx") as f:
    for file in f.namelist():
        if file.startswith("xl/media"):
            f.extract(file, path=unzip_path)