
__author__ = 'hongliangzhu'
__copyright__ = 'Copyright 2019, 朱宏梁'
__license__ = 'tjpu'

import os

import tensorflow as tf
from vgg import vgg16

from config import batch_size, testing_dir, training_dir
from image_helper import rgb_to_yuv, yuv_to_rgb
from read_input import init_file_path, get_dataset_iterator
from residual_encoder import ResidualEncoder
from config import debug, shuffle_buffer_size, image_size, image_resize_method
def read_image(filename):
    """
    Read and store image with RGB color space.
    :param filename_queue: the filename queue for image files
    :return: image with RGB color space
    """
    # Read image file
    content = tf.read_file(filename)
    # Decode the image with RGB color space
    rgb_image = tf.image.decode_jpeg(content, channels=3, name="color_image_original")
    # Resize image to the right image_size
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=image_resize_method)
    # Map all pixel element value into [0, 1]
    return tf.clip_by_value(tf.div(tf.cast(rgb_image, tf.float32), 255), 0.0, 1.0, name="color_image_in_0_1")

def create_folder(folder_path):
    """
    Create folder if not exist.
    :param folder_path:
    :return: None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def init_model(train=True):
    """
    Init model for both training and testing.
    :param train: indicate if current is in training
    :return: all stuffs that need for this model
    """
    # Create training summary folder if not exist
    create_folder("summary/train/images")

    # Create testing summary folder if not exist
    create_folder("summary/test/images")

    # Use gpu if exist
    with tf.device('/device:GPU:0'):
        # Init image data file path
        testdir = input("输入图片的路径：")

        #file_paths = init_file_path(testing_dir)

        # Init training flag and global step
        print("⏳ Init placeholder and variables...")
        is_training = tf.placeholder(tf.bool, name="is_training")
        global_step = tf.train.get_or_create_global_step()

        # Load vgg16 model
        print("🤖 Load vgg16 model...")
        vgg = vgg16.Vgg16()

        # Build residual encoder model
        print("🤖 Build residual encoder model...")
        residual_encoder = ResidualEncoder()

        # Get dataset iterator
        #iterator = get_dataset_iterator(file_paths, batch_size, shuffle=True)

        # Get color image
        #  color_image_rgb = iterator.get_next(name="color_image_rgb") # 获取下一张彩色图片
        color_image_rgb = read_image(testdir)
        color_image_yuv = rgb_to_yuv(color_image_rgb, "color_image_yuv") # 将获取的rgb图转换成yuv格式

        # Get gray image
        gray_image_one_channel = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image_one_channel") # 获取灰度图片
        # 由上一步得到的灰度图转换成rgb3通道格式
        gray_image_three_channels = tf.image.grayscale_to_rgb(gray_image_one_channel, name="gray_image_three_channels") # 三通道的灰度图
        gray_image_yuv = rgb_to_yuv(gray_image_three_channels, "gray_image_yuv") # 灰度图转换yuv格式的灰度图像

        # Build vgg model
        with tf.name_scope("vgg16"):
            vgg.build(gray_image_three_channels) #建立vgg模型

        # Predict model
        # 建立残差编码模型： input_data ：给第一层输入的数据   vgg： vgg模型    is_training： 一个标志指示是否在训练
        predict = residual_encoder.build(input_data=gray_image_three_channels, vgg=vgg, is_training=is_training) # 预测的u，v两个空间
        predict_yuv = tf.concat(axis=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")# 将y，u，v三个空间拼接
        predict_rgb = yuv_to_rgb(predict_yuv, "predict_rgb")

        # Get loss
        # 预测出来的uv两个通道与真实的uv两个通道的loss
        loss = residual_encoder.get_loss(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))

        # Prepare optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # global_step记录的其实是train阶段每一步的索引，或者说是训练迭代的计数器，比如说在最后画loss和 accuracy的横坐标即是global_step
            lr = tf.train.exponential_decay(0.001, global_step, 1000, 0.96)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step, name='optimizer')#global_step在训练中是计数的作用，每训练一个batch就加1

        # Init tensorflow summaries
        print("⏳ Init tensorflow summaries...")
        tf.summary.histogram("loss", loss)
        tf.summary.image("gray_image", gray_image_three_channels, max_outputs=1)
        tf.summary.image("predict_image", predict_rgb, max_outputs=1)
        tf.summary.image("color_image", color_image_rgb, max_outputs=1)

    return is_training, global_step, optimizer, loss, predict_rgb, color_image_rgb, gray_image_three_channels, testdir
