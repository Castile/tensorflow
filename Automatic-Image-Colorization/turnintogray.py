import os
import tensorflow as tf
from config import image_size, image_resize_method
from image_helper import yuv_to_rgb, rgb_to_yuv
import matplotlib.pyplot as plt
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

dir = input("请输入图片路径")
dir = str(dir)
print(dir)
color_image_rgb = read_image(dir)
plt.show(color_image_rgb)
color_image_yuv = rgb_to_yuv(color_image_rgb, "color_image_yuv") # 将获取的rgb图转换成yuv格式
# Get gray image
gray_image_one_channel = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image_one_channel") # 获取灰度图片


plt.imsave(gray_image_one_channel)
