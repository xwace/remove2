from abc import ABC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy


#from readtxt import txt2array

def txt2array(txt_path):
    with open(txt_path) as file:
        txt_file = file.readlines()
    img_height = len(txt_file)
    img_width = len(txt_file[0]) - 1
    temp_array = np.zeros((img_height, img_width), dtype="uint8")

    for i, r in enumerate(txt_file):
        if i >= img_height:
            continue
        r = r.strip()
        for j, c in enumerate(r):
            if j >= img_width:
                continue

            try:
                temp_array[i, j] = int(c)
            except BaseException as bs:
                print("c", i, ", ", j)


    return temp_array

def array2txt(temp_array, txt_path):
    f = open(txt_path, mode='w')
    h, w = temp_array.shape
    for i in range(h):
        for j in range(w):
            f.write(str(temp_array[i,j]))
        f.write("\n")

    f.close()

import torchvision.models.resnet as modules
if __name__ =="__main__":
    img_array = txt2array("map.txt")
    # txt_path = 'map.txt'	# txt文本路径
    # f = open(txt_path)
    # data_lists = f.readlines()	#读出的是str类型

    # dataset= []
	# # 对每一行作循环
    # for data in data_lists:
    #  data1 = data.strip('\n')	# 去掉开头和结尾的换行符
    #  data2 = data1.split('\t')	# 把tab作为间隔符
    #  dataset.append(data2)	# 把这一行的结果作为元素加入列表dataset

    # img_array = np.array(dataset)

    lut = np.zeros((1, 256), dtype="uint8")
    lut[0,0] = 1

    temp_list5 = Image.fromarray(img_array)
    plt.figure(1)
    plt.imshow(temp_list5)

    img_zero = cv2.LUT(img_array, lut)
    # 获取到的轮廓点是逆时针排序的
    contours, hierarchy = cv2.findContours(img_zero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lut[0, 0] = 0
    lut[0, 2] = 1
    img_two = cv2.LUT(img_array, lut)

    temp_list4 = Image.fromarray(img_two)
    plt.figure(2)
    plt.imshow(temp_list4)

    max_id = 0
    max_area = 0
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_id = i

    mask = np.zeros_like(img_zero)
    mask = cv2.drawContours(mask, contours, max_id, (1,), -1)

    temp_list2 = Image.fromarray(mask)
    plt.figure(3)
    plt.imshow(temp_list2)

    kernel = np.ones((3, 3), dtype="uint8")  # 不能太大，需要保留原有的轮廓形状特征
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    temp_list3 = Image.fromarray(mask)
    plt.figure(4)
    plt.imshow(temp_list3)

    mask_temp = cv2.bitwise_and(img_two, mask)

    img_array -= mask_temp*2

    temp_list = Image.fromarray(mask_temp)
    temp_list1 = Image.fromarray(img_array)
    plt.figure(5)
    plt.imshow(temp_list)
    plt.figure(6)
    plt.imshow(temp_list1)
    plt.show()
