#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""配置文件，包含所有的配置变量"""
__author__ = ',Hongliang Zhu <hongliangzhu2019@gmail.com>'
__copyright__ = 'Copyright 2019,Hongliang Zhu'

import numpy as np
import tensorflow as tf


# Debug flag, if true, will check model shape using assert in each step and skip gray image check part (to save time)
debug = False

# 训练图片的大小
image_size = 224

image_resize_method = tf.image.ResizeMethod.BILINEAR  # 双线性插值

# 神经网络的一些超参数
training_iters = 60000
batch_size = 6
display_step = 50  # 每50次打印损失信息并且保存日志
testing_step = 1000  # 每1000次测试效果
saving_step = 5000  # 每5000次保存一次模型
shuffle_buffer_size = 1000

# UV channel normalization parameters
u_norm_para = 0.435912
v_norm_para = 0.614777

# training_dir = "train2014"
# testing_dir = "test2014"
# training_dir = "trainimg"

# 训练集合测试集的目录
training_dir = "ILSVRC2012_img_test"
testing_dir = "testimg"

# 网络模型，生成的结果和图像存放的目录
summary_path = "summary"
training_summary = summary_path + "/train"
testing_summary = summary_path + "/test"


"""
 tf.truncated_normal:从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正态分布，
 均值和标准差自己设定。这是一个截断的产生正态分布的函数，就是说产生正态分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
"""

# 每一层的权重初始化（可训练）
weights = {
    'b_conv4': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01), trainable=True),
    'b_conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01), trainable=True),
    'b_conv2': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01), trainable=True),
    'b_conv1': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01), trainable=True),
    'b_conv0': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01), trainable=True),
    'output_conv': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01), trainable=True),
}

# 高斯模糊核 (不可训练参数)
gaussin_blur_3x3 = np.divide([
    [1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.],
], 16.) # (3, 3)  除法操作
gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2)  axis=-1也就是在最后一维后增加一维
gaussin_blur_3x3 = np.stack((gaussin_blur_3x3, gaussin_blur_3x3), axis=-1) # (3, 3, 2, 2)   对指定axis增加维度，

gaussin_blur_5x5 = np.divide([
    [1.,  4.,  7.,  4., 1.],
    [4., 16., 26., 16., 4.],
    [7., 26., 41., 26., 7.],
    [4., 16., 26., 16., 4.],
    [1.,  4.,  7.,  4., 1.],
], 273.) # (5, 5)
gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2)
gaussin_blur_5x5 = np.stack((gaussin_blur_5x5, gaussin_blur_5x5), axis=-1) # (5, 5, 2, 2)

# 将高斯核转换成tensorflow张量形式
tf_blur_3x3 = tf.Variable(tf.convert_to_tensor(gaussin_blur_3x3, dtype=tf.float32), trainable=False)
tf_blur_5x5 = tf.Variable(tf.convert_to_tensor(gaussin_blur_5x5, dtype=tf.float32), trainable=False)
