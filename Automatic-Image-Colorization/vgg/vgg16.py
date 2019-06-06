from __future__ import print_function # 使用python3的print语法 ，兼容
from __future__ import division

import inspect # 获取加载模块路径
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68] # VGG在所有训练图片的三个通道的均值


class Vgg16:
    def __init__(self, vgg16_npy_path=None): # vgg16_npy_path是VGG的预训练模型路径
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)  # 获取Vgg16该类的路径（本类）
            path = os.path.abspath(os.path.join(path, os.pardir))  # os.path.abspath(path) 返回绝对路径，os.pardir为当前目录的父目录
            path = os.path.join(path, "vgg16.npy")  # "vgg16.npy"文件名和路径合并成新路径
            vgg16_npy_path = path
            print(path) # 打印vgg16模型的路径

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        """   
        numpy将数组以二进制格式保存到磁盘
        np.load和np.save是读写磁盘数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中。
        np.save("A.npy",A)   #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。
        B=np.load("A.npy")        
        item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
        person={'name':'lizhong','age':'26','city':'BeiJing','blog':'www.jb51.net'}    
        for x in person.items():
            print x
        显示：[（'name'，'lizhong'） etc ]
        """

        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG 从npy文件中加载变量来建立VGG模型

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
                    rgb图片   [批， 高， 宽， 3通道] 它们的值在到0-1之间
        """

        start_time = time.time() # 加载开始时间
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(value=rgb_scaled, num_or_size_splits=3, axis=3)
        """
            tf.split: 把一个张量划分成几个子张量
                value：准备切分的张量 
                num_or_size_splits：准备切成几份 
                axis : 准备在第几个维度上进行切割 
        """
        #  x.get_shape()，只有tensor(张量)才可以使用这种方法，返回的是一个元组。
        #  x.get_shape().as_list())     #返回的元组重新变回一个列表
        assert red.get_shape().as_list()[1:] == [224, 224, 1]  # [1:]，第二个维度以后的所有维度, 维度为[224, 224, 1]，如果代表图像的话应该是[height,width,channel].
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    """
        max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
            value: 输入矩阵，四维。
            ksize：过滤器（池化核）的尺寸。是一个长度为4的以为数组，但是这个数组的一个数和最后一个数必须为1
            strides：步长。这个数组的一个数和最后一个数必须为1
            
    """
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)  # 获取权重 filter

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            """ conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None)
                    input: 输入矩阵，必须是4维度矩阵，在此处是bgr输入（[批， 高， 宽， 3通道]）
                    filter：当前卷积层使用的卷积核。这也是一个四维列表[1, 1, 1, 1]，前两个数值表示卷积核的大小1x1,第三个数值是输入图像的深度，第四个数值是本层过滤器的深度
                    strides：是一个长度为4的以为数组，但是这个数组的一个数和最后一个数必须为1。
                            比如[1, 2, 2, 1]表示长方向移动的步长为2，宽方向移动的步长也为2
            """

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)  # 将conv_biases加到此卷积层中
            """
            bias_add(value, bias, data_format=None, name=None):
                Adds `bias` to `value`.
            """
            relu = tf.nn.relu(bias)  # 添加激活函数
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    """
        #模型文件（.npy）部分内容如下：由一个字典组成，字典中的每一个键对应一层网络模型参数。（包括权重w和偏置b）
        a = {'conv1':[array([[1,2],[3,4]],dtype=float32),array([5,6],dtype=float32)],'conv2':[array([[1,2],[3,4]],dtype=float32),array([5,6],dtype=float32)]}

        conv1_w = a['conv1'][0]
        conv1_b = a['conv1'][1]
        conv2_w = a['conv2'][0]
        conv2_b = a['conv2'][1]
        --------------------- 
        
    """
    def get_conv_filter(self, name): # 得到权重W(四维张量)---vgg16.npy(通过"conv1_1"取出W，也就是filter)
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):  # 得到偏差b--vgg16.npy(通过"conv1_1"[1]取出b，也就是bias)
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
