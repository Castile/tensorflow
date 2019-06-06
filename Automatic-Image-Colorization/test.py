#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""测试模型"""

__author__ = ',Hongliang Zhu <hongliangzhu2019@gmail.com>'
__copyright__ = 'Copyright 2019,Hongliang Zhu'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import batch_size, display_step, saving_step, summary_path, testing_summary
from common import init_model
from image_helper import concat_images
import time

if __name__ == '__main__':
    # Init model
    is_training, _, _, loss, predict_rgb, color_image_rgb, gray_image, file_paths = init_model(train=False)

    # Init scaffold, hooks and config 分布式训练
    scaffold = tf.train.Scaffold()
    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=summary_path, save_steps=saving_step, scaffold=scaffold)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=(tf.GPUOptions(allow_growth=True)))
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, config=config, checkpoint_dir=summary_path)
    # Create a session for running operations in the Graph
    # 创建一个会话来运行计算图的操作
    with tf.train.MonitoredSession(session_creator=session_creator, hooks=[checkpoint_hook]) as sess:
        print("🤖 Start testing...")
        step = 0
        avg_loss = 0

        while not sess.should_stop():
            step += 1

            l, pred, color, gray = sess.run([loss, predict_rgb, color_image_rgb, gray_image], feed_dict={is_training: False})

            # Print batch loss
            print("📖 Testing iter %d, Minibatch Loss = %f" % (step, l))
            avg_loss += l

            # 保存所有的测试图片
            for i in range(len(color)):
                summary_image = concat_images(gray[i], pred[i])
                # summary_image = concat_images(summary_image, color[i])
                plt.imsave("%s/images/%d_%d.png" % (testing_summary, step, i), summary_image)
                # plt.imsave("%s/images/gray/%d_%d.jpeg" % (testing_summary, step, i), gray[i])
            if step >= len(file_paths) / batch_size:
                break

        print("🎉 Testing finished!")
        print("👀 Total average loss: %f" % (avg_loss / len(file_paths)))
