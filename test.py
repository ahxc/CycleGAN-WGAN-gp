from __future__ import print_function

from datetime import datetime
from random import shuffle

import os
import sys
import glob

import tensorflow as tf
import numpy as np

import cv2

from networks import *

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("X", './testset/man2woman/woman/', "path of x test datas.")# x域的测试图片路径
tf.flags.DEFINE_string("Y", './testset/man2woman/man/', "path of y test datas.")# y域的测试图片路径
tf.flags.DEFINE_string("out_X2Y", './testset/man2woman/woman/', "Output Folder")# 保存x到y图片的路径
tf.flags.DEFINE_string("out_Y2X", './testset/man2woman/man/', "Output Folder")# 保存y到x图片的路径
tf.flags.DEFINE_string("model_path", None, "Path of Snapshots")# 读取训练好的模型参数的路径

tf.flags.DEFINE_integer("image_size", 256, "load image size")# 输入的图像尺度

# 测试数据预处理
def TestImageReader(file_list, step, size):
    file_length = len(file_list)# 获取图片列表总长度

    line_idx = step % file_length# 获取一张待读取图片的下标
    test_line_content = file_list[line_idx]# 获取一张测试图片路径与名称

    # os.path.basename：返回最后的文件名，如果以/结尾则为空
    # os.path.splitext：分离文件名和扩展名
    test_image_name, _ = os.path.splitext(os.path.basename(test_line_content))# 获取该张测试图片名
    test_image = cv2.imread(test_line_content, 1)# 读取一张测试图片
    test_image = cv2.resize(test_image, (size, size)) / 127.5 - 1# 改变读取的测试图片的大小并归一化测试图片

    return test_image_name, test_image #返回读取并处理的一张测试图片与它的名称

def make_test_data_list(x_data_path, y_data_path):# make_test_data_list函数得到测试中的x域和y域的图像路径名称列表
    x_input_images = glob.glob(os.path.join(x_data_path, "*"))# 读取全部的x域图像路径名称列表
    y_input_images = glob.glob(os.path.join(y_data_path, "*"))# 读取全部的y域图像路径名称列表
    
    return x_input_images, y_input_images
 
def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    
    return img_rgb.astype(np.float32)# bgr
 
def main():
    if not os.path.exists(FLAGS.out_X2Y):# 如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(FLAGS.out_X2Y)
    if not os.path.exists(FLAGS.out_Y2X):# 如果保存y域测试结果的文件夹不存在则创建
        os.makedirs(FLAGS.out_Y2X)
 
    x_datalists, y_datalists = make_test_data_list(FLAGS.X, FLAGS.Y)# 得到待测试的x域和y域图像路径名称列表

    # 输入占位
    test_x_image = tf.placeholder(tf.float32,shape=[1, FLAGS.image_size, FLAGS.image_size, 3], name = 'test_x_image')
    test_y_image = tf.placeholder(tf.float32,shape=[1, FLAGS.image_size, FLAGS.image_size, 3], name = 'test_y_image')
 
    fake_y = generator(image = test_x_image, reuse = False, name = 'generator_x2y') #得到生成的y域图像
    fake_x = generator(image = test_y_image, reuse = False, name = 'generator_y2x') #得到生成的x域图像
 
    # 需要载入的已训练的模型参数
    restore_var = [v for v in tf.global_variables() if 'generator' in v.name]
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True# 设定显存不超量使用
    sess = tf.Session(config=config)# 建立会话层

    saver = tf.train.Saver(var_list = restore_var, max_to_keep=1)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))# 导入模型参数用以生成图像
 
    total_step = len(x_datalists) if len(x_datalists) > len(y_datalists) else len(y_datalists)# 测试的总步数及最大的图像个数
    for step in range(total_step):
        test_ximage_name, test_ximage = TestImageReader(x_datalists, step, FLAGS.image_size)# 得到x域的名称与输入图像
        test_yimage_name, test_yimage = TestImageReader(y_datalists, step, FLAGS.image_size)# 得到y域的名称与输入图像

        batch_x_image = np.expand_dims(np.array(test_ximage).astype(np.float32), axis = 0)# 填充维度，在axis = 0上增加1维(批量尺寸)
        batch_y_image = np.expand_dims(np.array(test_yimage).astype(np.float32), axis = 0)# 填充维度，在axis = 0上增加1维

        feed_dict = { test_x_image : batch_x_image, test_y_image : batch_y_image}# 建立feed_dict
        fake_x2y, fake_y2x = sess.run([fake_y, fake_x], feed_dict = feed_dict)# 得到生成的y域图像与x域图像

        wirte_x2y, wirte_y2x = cv_inv_proc(fake_x2y[0]), cv_inv_proc(fake_y2x[0])# 得到最终的图片结果
        x_write_image_name = FLAGS.out_X2Y + "/"+ test_ximage_name + "-1.png"# 待保存的x域图像与其对应的y域生成结果名字
        y_write_image_name = FLAGS.out_Y2X + "/"+ test_yimage_name + "-1.png"# 待保存的y域图像与其对应的x域生成结果名字

        cv2.imwrite(x_write_image_name, wirte_x2y)# 保存图像
        cv2.imwrite(y_write_image_name, wirte_y2x)# 保存图像
        print('step {:d}'.format(step))
 
if __name__ == '__main__':
    main()