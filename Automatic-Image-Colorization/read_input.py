#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""一些读取输入的函数定义"""

__author__ = ',Hongliang Zhu <hongliangzhu2019@gmail.com>'
__copyright__ = 'Copyright 2019,Hongliang Zhu'

import os
import cv2
import imghdr
import tensorflow as tf

from config import debug, shuffle_buffer_size, image_size, image_resize_method

def init_file_path(directory):
    paths = []
    for file_name in os.listdir(directory):  # 将图片路径下的所有图片取出
        file_path = '%s/%s' % (directory, file_name)  # 拼接成每个图片的路径
        paths.append(file_path)  # 将所有的jpeg格式的彩色图片添加到path数组里面
    return paths

# def init_file_path(directory):
#     """
#     获取图片路径的数组
#     :param directory: 图片存放的目录
#     :return: 返回一个数组，里面存放了每张彩色图片的路径
#     """
#     paths = []
#
#     if not debug:
#         print("Throwing all gray space images now... (this will take a long time if the training dataset is huge)")
#     # 将所有黑白图片删除
#     for file_name in os.listdir(directory): # 将图片路径下的所有图片取出
#         # Skip files that is not jpg
#         file_path = '%s/%s' % (directory, file_name) # 拼接成每个图片的路径
#         # imghdr是一个用来检测图片类型的模块，传递给它的可以是一个文件对象，也可以是一个字节流。
#         if imghdr.what(file_path) is not 'jpeg': # 不是jpeg格式的的图片直接跳过
#             continue
#         if not debug:
#             # 删除所有的灰度图片，以免面干扰训练效果
#             is_gray_space = True
#             img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED) # 读取这张图片
#             if len(img.shape) == 3 and img.shape[2] == 3: # 检查图片的维度，一般是3维度(宽*高*通道)，并且图片的第三个维度是颜色通道数，彩色图像是3个通道RGB
#                 for w in range(img.shape[0]):  # 图片宽度
#                     for h in range(img.shape[1]): # 图片的高度
#                         r, g, b = img[w][h] # 取出图片对应像素位置的三个颜色通道的值
#                         if r != g != b: # 如果三个颜色通道的值不相等，则不是灰度图片
#                             is_gray_space = False
#                         if not is_gray_space:  # 不是灰度图，终止内层循环
#                             break
#                     if not is_gray_space:  # 不是灰度图，终止内层循环
#                         break
#             if is_gray_space: # 如果是灰度图的话，执行以下删除操作，因为可能会出现异常，因此使用try语句
#                 try:
#                     os.remove(file_path)
#                 except OSError as e:
#                     print ("Error: %s - %s." % (e.filename, e.strerror))
#                 continue # 继续取出下一张图片进行判断是否为灰度图
#         paths.append(file_path) # 将所有的jpeg格式的彩色图片添加到path数组里面
#
#     return paths


def read_image(filename):
    """
    读取图片（rgb颜色空间）
    :param filename_queue: the filename queue for image files
    :return: 返回图片的rgb颜色空间
    """
    # 读取图片
    content = tf.read_file(filename)
    # 将图片编码为rgb的颜色空间
    rgb_image = tf.image.decode_jpeg(content, channels=3, name="color_image_original")
    # 将图片转换成正确的大小（224x224）
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=image_resize_method)
    # 将每个像素值都归一化到0-1之间
    return tf.clip_by_value(tf.div(tf.cast(rgb_image, tf.float32), 255), 0.0, 1.0, name="color_image_in_0_1")  # cast是类型转换


def get_dataset_iterator(filenames, batch_size, num_epochs=None, shuffle=False):
    """
    获取数据集迭代器:可以获取一个批次数量的图片
    :param filenames: 文件名
    :param batch_size: batch size
    :param num_epochs: number of epochs for producing each string before generating an OutOfRange error
    :param shuffle: if true, the strings are randomly shuffled within each epoch
    :return: the batch image data iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(read_image)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(count=num_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator
