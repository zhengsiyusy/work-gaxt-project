#!/usr/bin/python
# / -*-codeing = utf-8  -*-
# TIME : 2022/9/13 8:19
# File : read_excel
import os
import xlrd2
import cv2
from zipfile import ZipFile
from detect import detect

print_flag = 1


# 从excel中读取图片保存到media文件夹中
def save_image_in_excel():
    unzip_path = "./unzip"
    if not os.path.exists(unzip_path):
        os.mkdir(unzip_path)
    with ZipFile("image/1662623717826.xlsx") as f:
        for file in f.namelist():
            if file.startswith("xl/media"):
                f.extract(file, path=unzip_path)


def get_image_number_in_excel():
    global print_flag
    # 打开excel表格
    data_excel = xlrd2.open_workbook('image/1662623717826.xlsx')
    # 获取所有sheet名称
    names = data_excel.sheet_names()

    temp_table = data_excel.sheet_by_name(sheet_name='图片数据')  # 通过名称获取
    # excel工作表的行列操作
    n_cols = temp_table.ncols  # 获取该sheet中的有效列数
    n_rows = temp_table.nrows  # 获取该sheet中的有效行数
    if print_flag:
        print('列数', n_cols)
        print('行数', n_rows)
    return n_rows, temp_table


def read_hs_day_in_excel(col):
    n_rows, table = get_image_number_in_excel()
    # excel工作表的单元格操作
    if col != 0:
        # 返回单元格中的数据(第一列的所有数据)
        row_col_data = table.cell_value(rowx=col, colx=1)
        row_col_data_new = row_col_data.replace('"', '')
        row_col_data_new = row_col_data_new.replace(']', '')
        row_col_data_new = row_col_data_new.replace('[', '')
        row_col_data_new = row_col_data_new.replace('{', '')
        row_col_data_new = row_col_data_new.replace('}', '')
        # 以字符','将字符串分割开
        split_row_col_data = row_col_data_new.split(",")
        # print(split_row_col_data)
        if len(split_row_col_data) > 12:
            temp1 = split_row_col_data[7]
            temp_hs_data1 = temp1.replace('label:', '')
            temp_confidence = split_row_col_data[8]
            temp_confidence1 = temp_confidence.replace('prob:', '')
            confidence1 = float(temp_confidence1)
            temp2 = split_row_col_data[13]
            temp_hs_data2 = temp2.replace('label:', '')
            temp_confidence_sec = split_row_col_data[14]
            temp_confidence2 = temp_confidence_sec.replace('prob:', '')
            confidence2 = float(temp_confidence2)
            if confidence1 > confidence2:
                confidence = confidence1
                hs_data = temp_hs_data1
            else:
                confidence = confidence2
                hs_data = temp_hs_data2
            print("核酸天数：", hs_data)
            print('置信度：', confidence)
        else:
            temp_hs_data = split_row_col_data[7]
            hs_data = temp_hs_data.replace('label:', '')
            temp_confidence = split_row_col_data[8]
            temp_confidence1 = temp_confidence.replace('prob:', '')
            confidence = float(temp_confidence1)
            print("表格读取核酸天数：", hs_data)
            print('表格读取置信度：', confidence)
        return hs_data, confidence


if __name__ == '__main__':
    # 保存图片到unzip文件夹
    save_image_in_excel()

    rows, table = get_image_number_in_excel()
    print_flag = 0
    for i in range(rows-1):
        print("图", i+1, ":")
        excel_hs_data, excel_confidence = read_hs_day_in_excel(i + 1)
        fileName = "image" + str(i+1) + ".jpeg"
        path = "D:/YOLOV5/file_image_read/unzip/xl/media/" + fileName
        img = cv2.imread(path)
        # 传入一张图片
        detections = detect(img)
        for re_info in detections:
            yolov5_hs_data = re_info.get('class')
            if yolov5_hs_data != 'bjjkb':
                print("yolov5检测名称：", yolov5_hs_data)
                yolov5_confidence = re_info.get('conf')
                print("yolov5检测置信度：", yolov5_confidence)
                if yolov5_hs_data != excel_hs_data:
                    save_path = 'D:/YOLOV5/file_image_read/save_image/image'
                    cv2.imwrite(str(save_path) + str(i) + ".jpeg", img)




