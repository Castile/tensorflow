#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像预处理
图像操作的一些函数

"""

__author__ = ',Hongliang Zhu <hongliangzhu2019@gmail.com>'
__copyright__ = 'Copyright 2019,Hongliang Zhu'

import numpy as np
import tensorflow as tf
from config import u_norm_para, v_norm_para


def rgb_to_yuv(rgb_image, scope):
    """
    将图片的RGB颜色空间转换成YUV颜色空间
    :param rgb_image: rgb颜色空间的 图片
    :param scope: 函数的域名范围
    :return: 返回yuv颜色空间的图片
    """
    with tf.name_scope(scope):
        # 获取r，g，b颜色通道
        _r = tf.slice(rgb_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _g = tf.slice(rgb_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _b = tf.slice(rgb_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # 计算 y, u, v 通道
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _y = 0.299 * _r + 0.587 * _g + 0.114 * _b
        _u = 0.492 * (_b - _y)
        _v = 0.877 * (_r - _y)

        # 归一化 u, v 通道
        _u = _u / (u_norm_para * 2) + 0.5
        _v = _v / (v_norm_para * 2) + 0.5

        # 获取yuv颜色空间的图片
        return tf.clip_by_value(tf.concat(axis=3, values=[_y, _u, _v]), 0.0, 1.0)

def yuv_to_rgb(yuv_image, scope):
    """
     将图片的yuv颜色空间转换成rgb颜色空间
    :param yuv_image: yuv的图片
    :param scope: 函数的作用范围
    :return: rgb颜色空间
    """
    with tf.name_scope(scope):
        # 获取 y, u, v 通道
        _y = tf.slice(yuv_image, [0, 0, 0, 0], [-1, -1, -1, 1])
        _u = tf.slice(yuv_image, [0, 0, 0, 1], [-1, -1, -1, 1])
        _v = tf.slice(yuv_image, [0, 0, 0, 2], [-1, -1, -1, 1])

        # 去归一化 u, v channel
        _u = (_u - 0.5) * u_norm_para * 2
        _v = (_v - 0.5) * v_norm_para * 2

        # 计算 r, g, b 通道
        # https://www.pcmag.com/encyclopedia/term/55166/yuv-rgb-conversion-formulas
        _r = _y + 1.14 * _v
        _g = _y - 0.395 * _u - 0.581 * _v
        _b = _y + 2.033 * _u

        # 获取rgb颜色空间的图片
        return tf.clip_by_value(tf.concat(axis=3, values=[_r, _g, _b]), 0.0, 1.0)

def concat_images(img_a, img_b):
    """
    并排组合两个彩色图像。  后面可视化需要用到
    :param img_a:左边的彩色图片
    :param img_b: 右边的彩色图片
    :return: 合并后的图片
    """
    height_a, width_a = img_a.shape[:2] # 图片a的高和宽
    height_b, width_b = img_b.shape[:2]
    max_height = np.max([height_a, height_b])
    total_width = width_a + width_b # 宽度是两张图片的宽度和
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32) # 初始化合并后的新图片
    new_img[:height_a, :width_a] = img_a
    new_img[:height_b, width_a:total_width] = img_b
    return new_img
