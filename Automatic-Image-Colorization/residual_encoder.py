#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
残差编码模型的实现

参考：http://tinyclouds.org/colorize/
"""

__author__ = ',Hongliang Zhu <hongliangzhu2019@gmail.com>'
__copyright__ = 'Copyright 2019,Hongliang Zhu'

import cv2
import tensorflow as tf

from config import batch_size, debug, weights, image_resize_method, tf_blur_3x3, tf_blur_5x5


class ResidualEncoder(object):
    @staticmethod
    def get_weight(scope):
        """
        为特定的层初始化权重
        :param scope: 网络层的名称
        :return: 返回该层的权重值
        """
        return weights[scope]

    @staticmethod
    def get_loss(predict_val, real_val): # 损失函数
        """
        Loss function.
        :param predict_val: 预测值  u v
        :param real_val: 真实值
        :return: loss
        """
        if debug:
            assert predict_val.get_shape().as_list()[1:] == [224, 224, 2]
            assert real_val.get_shape().as_list()[1:] == [224, 224, 2]

        blur_real_3x3 = tf.nn.conv2d(real_val, tf_blur_3x3, strides=[1, 1, 1, 1], padding='SAME', name="blur_real_3x3")
        blur_real_5x5 = tf.nn.conv2d(real_val, tf_blur_5x5, strides=[1, 1, 1, 1], padding='SAME', name="blur_real_5x5")
        blur_predict_3x3 = tf.nn.conv2d(predict_val, tf_blur_3x3, strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_3x3")
        blur_predict_5x5 = tf.nn.conv2d(predict_val, tf_blur_5x5, strides=[1, 1, 1, 1], padding='SAME', name="blur_predict_5x5")

        diff_original = tf.reduce_sum(tf.squared_difference(predict_val, real_val), name="diff_original")  # 两个矩阵相减，然后对矩阵的每个元素求平方
        diff_blur_3x3 = tf.reduce_sum(tf.squared_difference(blur_predict_3x3, blur_real_3x3), name="diff_blur_3x3")
        diff_blur_5x5 = tf.reduce_sum(tf.squared_difference(blur_predict_5x5, blur_real_5x5), name="diff_blur_5x5")
        return (diff_original + diff_blur_3x3 + diff_blur_5x5) / 3

    @staticmethod
    def batch_normal(input_data, scope, is_training):
        """
            BN层
        """
        return tf.layers.batch_normalization(input_data, training=is_training, name=scope)

    def conv_layer(self, layer_input, scope, is_training, relu=True, bn=True):
        """
        卷积层的定义

        """
        with tf.variable_scope(scope):
            weight = self.get_weight(scope)
            output = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME', name="conv")
            if bn:
                output = self.batch_normal(output, is_training=is_training, scope=scope + '_bn')
            if relu:
                output = tf.nn.relu(output, name="relu")
            else:
                output = tf.sigmoid(output, name="sigmoid")
            return output

    def build(self, input_data, vgg, is_training):
        """
        建立残差编码模型
        :param input_data: 第一层的输入数据
        :param vgg: VGG 模型
        :param is_training:
        :return: None
        """
        if debug:
            # x.get_shape()，只有tensor才可以使用这种方法，返回的是一个元组。只能用于tensor来返回shape，但是是一个元组，需要通过as_list()的操作转换成list.
            assert input_data.get_shape().as_list()[1:] == [224, 224, 3]

        # Batch norm and 1x1 convolutional layer 4
        bn_4 = self.batch_normal(vgg.conv4_3, "bn_4", is_training)  # vgg网络的第4层经过BN处理
        b_conv4 = self.conv_layer(bn_4, "b_conv4", is_training, bn=False)  # 1x1卷积核由28x28x512---->28x28x256

        if debug:
            assert bn_4.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and add convolutional layer 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56], method=image_resize_method) # 双线性插值(上采样)
        bn_3 = self.batch_normal(vgg.conv3_3, "bn_3", is_training) # 得到vgg网络的第三层并进行batchnorm操作
        b_conv3_input = tf.add(bn_3, b_conv4_upscale, name="b_conv3_input") # 残差编码网络的第三层的输入是由vgg网络第三层经过batchnorm操作后加上残差编码网络的第4层上采样后的输出
        b_conv3 = self.conv_layer(b_conv3_input, "b_conv3", is_training)  # 3x3卷积核操作由28x28x256--->56x56x128


        # Backward upscale layer 3 and add convolutional layer 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112], method=image_resize_method)
        bn_2 = self.batch_normal(vgg.conv2_2, "bn_2", is_training)
        b_conv2_input = tf.add(bn_2, b_conv3_upscale, name="b_conv2_input")
        b_conv2 = self.conv_layer(b_conv2_input, "b_conv2", is_training)

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert bn_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2_input.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and add convolutional layer 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224], method=image_resize_method)
        bn_1 = self.batch_normal(vgg.conv1_2, "bn_1", is_training)
        b_conv1_input = tf.add(bn_1, b_conv2_upscale, name="b_conv1_input")
        b_conv1 = self.conv_layer(b_conv1_input, "b_conv1", is_training)

        if debug:
            assert b_conv2_upscale.get_shape().as_list()[1:] == [224, 224, 64]
            assert bn_1.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1_input.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1.get_shape().as_list()[1:] == [224, 224, 3]

        # Backward upscale layer 1 and add input layer
        bn_0 = self.batch_normal(input_data, "bn_0", is_training)
        b_conv0_input = tf.add(bn_0, b_conv1, name="b_conv0_input")
        b_conv0 = self.conv_layer(b_conv0_input, "b_conv0", is_training)

        if debug:
            assert bn_0.get_shape().as_list()[1:] == [224, 224, 3]
            assert b_conv0_input.get_shape().as_list()[1:] == [224, 224, 3]
            assert b_conv0.get_shape().as_list()[1:] == [224, 224, 3]

        # Output layer
        output_layer = self.conv_layer(b_conv0, "output_conv", is_training, relu=False)

        if debug:
            assert output_layer.get_shape().as_list()[1:] == [224, 224, 2]

        return output_layer
