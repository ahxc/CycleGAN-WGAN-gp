import tensorflow as tf
import numpy as np
import os
import glob
import cv2
#-----------------------------------------------------------------------------------
# 图像读取
def imreader(x_list, y_list, step, size):
    length = len(x_list)

    line_idx = step%length
    x_path = x_list[line_idx]
    y_path = y_list[line_idx]

    x_image = cv2.imread(x_path, 1) 
    y_image = cv2.imread(y_path, 1)

    x_image = cv2.resize(x_image, (size, size))
    x_image = x_image/127.5-1.
    y_image = cv2.resize(y_image, (size, size))
    y_image = y_image/127.5-1.

    return x_image, y_image
#-----------------------------------------------------------------------------------
# 文件目录创建
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#-----------------------------------------------------------------------------------
# 模型存储
def save(saver, sess, logdir, step):
    model_name = 'model'
    checkpoint_path = os.path.join(logdir, model_name)

    check_dir(logdir)

    saver.save(sess, checkpoint_path, global_step = step)
    print('The checkpoint has been created.')
#-----------------------------------------------------------------------------------
# 还原归一化后的图片
def imrestore(img):
    img = (img+1.)*127.5

    return img.astype(np.float32)
#-----------------------------------------------------------------------------------
# 将训练过程中的结果结合位一张图片
def imget(x_image, y_image, G_realx_fakey, G_fakey_fakex, G_realy_fakex, G_fakex_fakey):
    x_image = imrestore(x_image)
    y_image = imrestore(y_image)

    G_realx_fakey = imrestore(G_realx_fakey[0])
    G_fakey_fakex = imrestore(G_fakey_fakex[0])
    G_realy_fakex = imrestore(G_realy_fakex[0])
    G_fakex_fakey = imrestore(G_fakex_fakey[0])

    row1 = np.concatenate((x_image, G_realx_fakey, G_fakey_fakex), axis=1)
    row2 = np.concatenate((y_image, G_realy_fakex, G_fakex_fakey), axis=1)

    return np.concatenate((row1, row2), axis=0)
#-----------------------------------------------------------------------------------
# 对齐两个域图片数量
def alignment_data(A, B):
    if len(A) == len(B):
        return A, B

    if len(A) < len(B):
        return A, sorted(B)[:len(A)]

    return A[:len(B)], B
#-----------------------------------------------------------------------------------
# 获取图片的路径列表
def make_train_data_list(x_data_path, y_data_path):
    x_input_images_raw = glob.glob(os.path.join(x_data_path, "*"))
    y_input_images_raw = glob.glob(os.path.join(y_data_path, "*"))
    
    return alignment_data(x_input_images_raw, y_input_images_raw)
#-----------------------------------------------------------------------------------
# 增加批次维度
def expand_dims(img):
    return np.expand_dims(np.array(img).astype(np.float32), axis=0)
#-----------------------------------------------------------------------------------