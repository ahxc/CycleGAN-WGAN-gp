from __future__ import print_function

from datetime import datetime
from random import shuffle
import random

import os
import sys

import tensorflow as tf
import numpy as np

import glob
import cv2

from networks import *

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "./model/", "path of model")# 保存模型的路径
tf.flags.DEFINE_string("load_model", None, "loading model")# 加载训练好的模型
tf.flags.DEFINE_string("X", "./dataset/man2woman/woman/", "path of x training datas")# x域的训练图片路径
tf.flags.DEFINE_string("Y", "./dataset/man2woman/man/", "path of y training datas.")# y域的训练图片路径
tf.flags.DEFINE_string("outfile", "./outfile/", "path of y training datas.")# y域的训练图片路径

tf.flags.DEFINE_integer("image_size", 256, "image size, default: 256")# 网络输入的尺度
tf.flags.DEFINE_integer("random_seed", 42, "random seed, default: 42")# 随机数种子
tf.flags.DEFINE_integer("sum_step", 50000, "all step, default: 50000")# 训练次数
tf.flags.DEFINE_integer("save_model_step", 10000, "times to save model")# 训练中每过多少step保存模型
tf.flags.DEFINE_integer("summary_step", 100, "times to summary")# 训练中每过多少step保存训练日志(记录一下loss值)
tf.flags.DEFINE_integer("outfile_step", 100, "times to outfile")# 训练中每过多少step保存可视化图像
tf.flags.DEFINE_integer("loss_step", 100, "times to outfile")# 训练中每过多少step打印训练损失

tf.flags.DEFINE_float("gp_lambda", 1, "Weight of gradient penalty term, default: 5")# 训练中梯度惩罚前的乘数
tf.flags.DEFINE_float("l1_lambda", 10, "Weight of l1_loss, default: 10")# 训练中L1_Loss前的乘数
tf.flags.DEFINE_float("learning_rate", 2e-4, "initial learning rate, default: 0.0002")# 基本训练学习率
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam, default: 0.5")# adam学习器的beta1参数

# 训练数据预处理
def TrainImageReader(x_file_list, y_file_list, step, size):
    file_length = len(x_file_list)# 获取图片名称列表总长度

    line_idx = step % file_length # 以数据集长度为周期获得获取一张图片的下标
    x_line_content = x_file_list[line_idx]# 获取一张x域图片路径与名称
    y_line_content = y_file_list[line_idx]# 获取一张y域图片路径与名称

    # <0以原图像读取，0以灰度图像读取，>0以RGB格式读取
    x_image = cv2.imread(x_line_content, 1)# 读取一张x域的图片 
    y_image = cv2.imread(y_line_content, 1)# 读取一张y域的图片

    x_image = cv2.resize(x_image, (size, size))# 改变读取的x域图片的大小
    x_image = x_image/127.5-1.# 归一化x域的图片
    y_image = cv2.resize(y_image, (size, size))# 改变读取的y域图片的大小
    y_image = y_image/127.5-1.# 归一化y域的图片

    #归一化
    # 255 范围[0, 1]
    # 127.5 - 1 范围[-1, 1]

    return x_image, y_image# 返回读取并处理的一张x域图片和y域图片

# 模型存储
def save(saver, sess, logdir, step):
    model_name = 'model'# 保存的模型名前缀
    checkpoint_path = os.path.join(logdir, model_name)# 模型的保存路径与名称

    if not os.path.exists(logdir):# 如果路径不存在即创建
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step = step)# 保存模型，依次是会话，保存路径，以步骤为保存编号
    print('The checkpoint has been created.')

def cv_inv_proc(img):# 将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5

    return img_rgb.astype(np.float32)

# 输出训练过程中的结果
def get_write_picture(x_image, y_image, G_realx_fakey, G_fakey_fakex, G_realy_fakex, G_fakex_fakey):
    x_image = cv_inv_proc(x_image)# 还原x域的图像
    y_image = cv_inv_proc(y_image)# 还原y域的图像

    G_realx_fakey = cv_inv_proc(G_realx_fakey[0])# 还原生成的y域的图像
    G_fakey_fakex = cv_inv_proc(G_fakey_fakex[0])# 还原重建的x域的图像
    G_realy_fakex = cv_inv_proc(G_realy_fakex[0])# 还原生成的x域的图像
    G_fakex_fakey = cv_inv_proc(G_fakex_fakey[0])# 还原重建的y域的图像

    row1 = np.concatenate((x_image, G_realx_fakey, G_fakey_fakex), axis=1)# 得到训练中可视化结果的第一行，将图像列轴(x)拼接(左右)
    row2 = np.concatenate((y_image, G_realy_fakex, G_fakex_fakey), axis=1)# 得到训练中可视化结果的第二行

    return np.concatenate((row1, row2), axis=0) #得到训练中可视化结果 #将图像行轴(y)拼接(上下)
 
# 定义对齐x域和y域的图像数量
def add_train_list(x_input_images_raw, y_input_images_raw):
    # 获得x域y域图片数量
    length_x, length_y = len(x_input_images_raw), len(y_input_images_raw)
    
    # 如果x域和y域图像数量本来就一致，直接返回打乱的列表结果
    if length_x == length_y:
        return shuffle(x_input_images_raw), shuffle(y_input_images_raw)

    # 如果x域的训练图像数量大于y域的训练图像数量，则随机选择y域的图像补充y域
    elif length_x > length_y:
        num = int(length_x - length_y)# 计算两域图像数量相差的个数

        y_append_images = []# 初始化待补充的名称列表
        for i in range(num):
            j = random.randint(0, length_y - 1)
            y_append_images.append(y_input_images_raw[j])
        y_input_images_raw += y_append_images# 将抽取的名称列表与待补充的列表合并

    else:
        num = int(length_y - length_x)

        x_append_images = []
        for i in range(num):
            j = random.randint(0, length_x - 1)
            x_append_images.append(x_input_images_raw[j])
        x_input_images_raw += x_append_images
    
    # 打乱名称列表顺序    
    shuffle(y_input_images_raw)
    shuffle(x_input_images_raw)

    return x_input_images_raw, y_input_images_raw

# 定义x域和y域的图像名称列表获取
def make_train_data_list(x_data_path, y_data_path):
    # 读取全部的x域图像路径名称列表
    x_input_images_raw = glob.glob(os.path.join(x_data_path, "*"))# *为通配符# glob.glob会加入所有文件名次，和共同的文件路径
    # 读取全部的y域图像路径名称列表
    y_input_images_raw = glob.glob(os.path.join(y_data_path, "*"))
    
    # 将x域图像数量与y域图像数量对齐
    return add_train_list(x_input_images_raw, y_input_images_raw)

def main():
    # 如果是加载模型则不创建
    if FLAGS.load_model is not None:
        model_path = FLAGS.load_model# 模型子文件夹路径
        outfile_path = os.path.join(FLAGS.outfile, model_path.split('/')[-1])# 可视化结果子文件夹路径
    # 否则创建本次训练的模型子文件夹
    else:
        # 以当前时间命名子文夹
        current_time = datetime.now().strftime("%Y%m%d-%H-%M")
        model_path = os.path.join(FLAGS.model_path, current_time)
        outfile_path = os.path.join(FLAGS.outfile, current_time)
        os.makedirs(model_path)
        os.makedirs(outfile_path)

    # 得到数量相同的x域和y域图像路径名称列表
    x_datalists, y_datalists = make_train_data_list(FLAGS.X, FLAGS.Y)

    tf.set_random_seed(FLAGS.random_seed)# 初始一下随机数

    # 输入占位
    x_img = tf.placeholder(tf.float32,shape = [1, FLAGS.image_size, FLAGS.image_size, 3], name = 'x_img')# x域输入
    y_img = tf.placeholder(tf.float32,shape = [1, FLAGS.image_size, FLAGS.image_size, 3], name = 'y_img')# y域输入
    lr = tf.placeholder(tf.float32, None, name = 'learning_rate')# 学习率
 
    G_realx_fakey = generator(image = x_img, reuse = False, name = 'generator_x2y')# 真x-假y
    G_fakey_fakex = generator(image = G_realx_fakey, reuse = False, name = 'generator_y2x')# 假y-假x
    G_realy_fakex = generator(image = y_img, reuse = True, name = 'generator_y2x')# 真y-假x
    G_fakex_fakey = generator(image = G_realy_fakex, reuse = True, name = 'generator_x2y')# 假x-真y
 
    dy_fake = discriminator(image = G_realx_fakey, reuse = False, name = 'discriminator_y')# 判别器返回的对生成的y域图像的判别结果
    dx_fake = discriminator(image = G_realy_fakex, reuse = False, name = 'discriminator_x')# 判别器返回的对生成的x域图像的判别结果
    dy_real = discriminator(image = y_img, reuse = True, name = 'discriminator_y')# 判别器返回的对真实的y域图像的判别结果
    dx_real = discriminator(image = x_img, reuse = True, name = 'discriminator_x')# 判别器返回的对真实的x域图像的判别结果

    # 计算生成器的损失
    gen_loss = single_loss(dx_fake) + single_loss(dy_fake) + FLAGS.l1_lambda * l1_loss(x_img, G_fakey_fakex) + FLAGS.l1_lambda * l1_loss(y_img, G_fakex_fakey)
 
    # y域判别器损失
    dy_loss = (single_loss(dy_real) + single_loss(dy_fake)) / 2# 计算判别器判别的y域图像的loss

    # y域梯度惩罚
    # dy_loss += gradient_penalty(y_img, G_realx_fakey, FLAGS.gp_lambda, reuse = True, name = 'discriminator_y')
 
    # x域判别器损失
    dx_loss = (single_loss(dx_real) + single_loss(dx_fake)) / 2# 计算判别器判别的x域图像的loss

    # x域梯度惩罚
    # dx_loss += gradient_penalty(x_img, G_realy_fakex, FLAGS.gp_lambda, reuse = True, name = 'discriminator_x')
 
    dis_loss = dy_loss + dx_loss# 所有判别器的loss
 
    # 记录生成器loss的日志
    gen_loss_sum = tf.summary.scalar("final_objective", gen_loss)
    
    # 记录判别器的loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)
    dx_loss_sum = tf.summary.scalar("dx_loss", dx_loss) #记录判别器判别的x域图像的loss的日志
    dy_loss_sum = tf.summary.scalar("dy_loss", dy_loss) #记录判别器判别的y域图像的loss的日志

    # 合并判别器日志
    discriminator_sum = tf.summary.merge([dx_loss_sum, dy_loss_sum, dis_loss_sum])
 
    # tf.trainable_variables返回可训练的变量list
    # if 为 v 的限制条件
    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]# 所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]# 所有判别器的可训练参数

    adam = tf.train.AdamOptimizer(lr, beta1 = FLAGS.beta1)# adam集成学习器
 
    d_grads_and_vars = adam.compute_gradients(dis_loss, var_list = d_vars)# 判别器训练梯度
    g_grads_and_vars = adam.compute_gradients(gen_loss, var_list = g_vars)# 生成器训练梯度

    d_train = adam.apply_gradients(d_grads_and_vars)# 更新判别器参数
    g_train = adam.apply_gradients(g_grads_and_vars)# 更新生成器参数

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True# 设定显存不超量使用
    sess = tf.Session(config = config)# 新建会话层

    train_op = tf.group(d_train, g_train)# train_op返回执行其所有的输入的操作(操作的聚合)
    summary_writer = tf.summary.FileWriter(model_path, graph=tf.get_default_graph())# 日志记录器
    saver = tf.train.Saver()# 模型保存器(放置在模型重载前面，会话层后面)

    # 模型重载
    if FLAGS.load_model is not None:
        # 获得获得模型统一的(无后缀名)所在路径model_checkpoint_path和all_model_checkpoint_paths
        checkpoint = tf.train.get_checkpoint_state(model_path)

        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"# 检查点文件

        restore = tf.train.import_meta_graph(meta_graph_path)# 导入图用以继续训练
        restore.restore(sess, tf.train.latest_checkpoint(model_path))# 重载最新模型
        step_now = int(meta_graph_path.split("-")[-1].split(".")[0])# 获得当前模型的步数

    else:
        init = tf.global_variables_initializer() #参数初始化器
        sess.run(init) #初始化所有可训练参数
        step_now = 0# 若不重载模型则设置从0开始训练

    lrate = FLAGS.learning_rate# 得到学习率

    # 如果当前步数为存储周期的倍数则+1，避免模型重载时重复存储模型
    step_now = step_now + 1 if step_now % FLAGS.save_model_step == 0 and step_now != 0 else step_now
    for step in range(step_now, FLAGS.sum_step):
        # 每隔一轮数据集周期打乱一下数据集
        if step % len(x_datalists) == 0:
            shuffle(x_datalists)
            shuffle(y_datalists)

        x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step, FLAGS.image_size)# 读取x域图像和y域图像
        batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0)# 在axis=0上增加一维(批量尺寸1)
        batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0)# 在axis=0上增加一维
        feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image}# 得到feed_dict
        gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op], feed_dict=feed_dict)# 得到每个step中的生成器和判别器loss

        if step % FLAGS.save_model_step == 0:# 每过save_model_step次保存模型
            save(saver, sess, model_path, step)

        if step % FLAGS.summary_step == 0:# 每过summary_step次保存训练日志
            gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, discriminator_sum], feed_dict = feed_dict)
            summary_writer.add_summary(gen_loss_sum_value, step)# G日志
            summary_writer.add_summary(discriminator_sum_value, step)# D日志

        if step % FLAGS.outfile_step == 0:# 每过outfile_step次写一下训练的可视化结果
            realx_fakey, fakey_fakex, realy_fakex, fakex_fakey = sess.run([G_realx_fakey, G_fakey_fakex, G_realy_fakex, G_fakex_fakey], feed_dict = feed_dict)# run出网络输出
            write_image = get_write_picture(x_image_resize, y_image_resize, realx_fakey, fakey_fakex, realy_fakex, fakex_fakey)# 得到训练的可视化结果
            write_image_name = os.path.join(outfile_path, "step-{}.png".format(str(step)))# 待保存的训练可视化结果路径与名称
            cv2.imwrite(write_image_name, write_image)# 保存训练的可视化结果

        if step % FLAGS.loss_step == 0:# 打印损失信息
            print('step {:d} \t all_G_loss = {:.3f}, all_D_loss = {:.3f}'.format(step, gen_loss_value, dis_loss_value))

if __name__ == '__main__':
    main()