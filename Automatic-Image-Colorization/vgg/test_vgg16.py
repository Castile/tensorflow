import numpy as np
import tensorflow as tf
import utils
from vgg import vgg16

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(
#         config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
#     images = tf.placeholder("float", [2, 224, 224, 3])
#     feed_dict = {images: batch}
#
#     vgg = vgg16.Vgg16()
#     with tf.name_scope("content_vgg"):
#         vgg.build(images)
#
#     prob = sess.run(vgg.prob, feed_dict=feed_dict)
#     print(prob)
#     utils.print_prob(prob[0], './synset.txt')
#     utils.print_prob(prob[1], './synset.txt')


with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()  ##引用vgg16.py里面定义的Vgg16类
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)  ### 在build()函数里面，属于成员变量，self.prob = tf.nn.softmax(self.fc8, name="prob")
        print(prob)  ####打印的预测矩阵（2,1000）类似手写字识别： [1,0,0,0...]:表示0
        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')

