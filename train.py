from datetime import datetime
from random import shuffle
from networks import *
from utils import *
from loss import *

import random
import os
import sys
import tensorflow as tf
import numpy as np
import glob
import cv2
#-----------------------------------------------------------------------------------
# 超参数入口
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "./model/", "path of model")
tf.flags.DEFINE_string("load_model", None, "loading model")
tf.flags.DEFINE_string("dataset", "./dataset/person2cartoon/", "path of training datas")
tf.flags.DEFINE_string("outfile", "./outfile/", "path of y training datas.")
tf.flags.DEFINE_string('GAN_type', 'GAN', 'GAN/LSGAN/WGAN')

tf.flags.DEFINE_integer("image_size", 256, "image size, default: 256")
tf.flags.DEFINE_integer("random_seed", 42, "random seed, default: 42")
tf.flags.DEFINE_integer("sum_step", 50000, "all step, default: 50000")
tf.flags.DEFINE_integer("save_model_step", 10000, "times to save model")
tf.flags.DEFINE_integer("summary_step", 100, "times to summary")
tf.flags.DEFINE_integer("outfile_step", 100, "times to outfile")
tf.flags.DEFINE_integer("loss_step", 100, "times to outfile")

tf.flags.DEFINE_float("gp_lambda", 5, "Weight of gradient penalty term, default: 5")
tf.flags.DEFINE_float("l1_lambda", 10, "Weight of l1_loss, default: 10")
tf.flags.DEFINE_float("learning_rate", 2e-4, "initial learning rate, default: 0.0002")
tf.flags.DEFINE_float("beta1", 0.5, "momentum term of adam, default: 0.5")
#-----------------------------------------------------------------------------------
# 主进程
def main():
    if FLAGS.load_model is not None:
        model_path = FLAGS.load_model
        outfile_path = os.path.join(FLAGS.outfile, model_path.split('/')[-1])

    else:
        current_time = datetime.now().strftime("%Y%m%d-%H-%M")
        model_path = os.path.join(FLAGS.model_path, current_time)
        outfile_path = os.path.join(FLAGS.outfile, current_time)

        check_dir(model_path)
        check_dir(outfile_path)

    X_path = os.path.join(FLAGS.dataset, 'trainA')
    Y_path = os.path.join(FLAGS.dataset, 'trainB')

    x_datalists, y_datalists = make_train_data_list(X_path, Y_path)

    tf.set_random_seed(FLAGS.random_seed)

    # 输入占位
    x_img = tf.placeholder(tf.float32, shape=[1, FLAGS.image_size, FLAGS.image_size, 3], name='x_img')
    y_img = tf.placeholder(tf.float32, shape=[1, FLAGS.image_size, FLAGS.image_size, 3], name='y_img')
    lr = tf.placeholder(tf.float32, None, name='learning_rate')
 
    G_realx_fakey = generator(X=x_img, reuse=False, scope='generator_x2y')
    G_fakey_fakex = generator(X=G_realx_fakey, reuse=False, scope='generator_y2x')
    G_realy_fakex = generator(X=y_img, reuse=True, scope='generator_y2x')
    G_fakex_fakey = generator(X=G_realy_fakex, reuse=True, scope='generator_x2y')
 
    dy_fake = discriminator(X=G_realx_fakey, reuse=False, scope='discriminator_y')
    dx_fake = discriminator(X=G_realy_fakex, reuse=False, scope='discriminator_x')
    dy_real = discriminator(X=y_img, reuse=True, scope='discriminator_y')
    dx_real = discriminator(X=x_img, reuse=True, scope='discriminator_x')

    # 定义损失
    gen_loss = g_loss(dx_fake, FLAGS.GAN_type)+g_loss(dy_fake, FLAGS.GAN_type)+\
    FLAGS.l1_lambda*l1_loss(x_img, G_fakey_fakex)+FLAGS.l1_lambda*l1_loss(y_img, G_fakex_fakey)

    dy_loss = d_loss(dy_real, dy_fake, FLAGS.GAN_type)
    dx_loss = d_loss(dx_real, dx_fake, FLAGS.GAN_type)

    if FLAGS.GAN_type == 'WGAN':
        dy_loss += gradient_penalty(y_img, G_realx_fakey, FLAGS.gp_lambda, reuse=True, scope='discriminator_y')
        dx_loss += gradient_penalty(x_img, G_realy_fakex, FLAGS.gp_lambda, reuse=True, scope='discriminator_x')

    dis_loss = dy_loss+dx_loss
 
    # 记录生成器loss日志
    gen_loss_sum = tf.summary.scalar("final_objective", gen_loss)

    # 记录判别器的loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)
    dx_loss_sum = tf.summary.scalar("dx_loss", dx_loss)
    dy_loss_sum = tf.summary.scalar("dy_loss", dy_loss)

    discriminator_sum = tf.summary.merge([dx_loss_sum, dy_loss_sum, dis_loss_sum])
 
    # tf.trainable_variables返回可训练的变量list
    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

    adam = tf.train.AdamOptimizer(lr, beta1 = FLAGS.beta1)
 
    d_grads_and_vars = adam.compute_gradients(dis_loss, var_list=d_vars)
    g_grads_and_vars = adam.compute_gradients(gen_loss, var_list=g_vars)

    d_train = adam.apply_gradients(d_grads_and_vars)
    g_train = adam.apply_gradients(g_grads_and_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)

    train_op = tf.group(d_train, g_train)
    summary_writer = tf.summary.FileWriter(model_path, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    # 模型重载
    if FLAGS.load_model is not None:
        checkpoint = tf.train.get_checkpoint_state(model_path)

        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"

        restore = tf.train.import_meta_graph(meta_graph_path)
        restore.restore(sess, tf.train.latest_checkpoint(model_path))
        step_now = int(meta_graph_path.split("-")[-1].split(".")[0])

    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        step_now = 0

    lrate = FLAGS.learning_rate# 得到学习率

    # 如果当前步数为存储周期的倍数则+1，避免模型重载时重复存储模型
    step_now = step_now + 1 if step_now % FLAGS.save_model_step == 0 and step_now != 0 else step_now
    for step in range(step_now, FLAGS.sum_step):
        # 每隔一轮数据集周期打乱一下数据集
        if step % len(x_datalists) == 0:
            shuffle(x_datalists)
            shuffle(y_datalists)

        x_image_resize, y_image_resize = imreader(x_datalists, y_datalists, step, FLAGS.image_size)

        batch_x_image = expand_dims(x_image_resize)
        batch_y_image = expand_dims(y_image_resize)

        feed_dict = { lr : lrate, x_img : batch_x_image, y_img : batch_y_image}

        gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op],
                                                     feed_dict=feed_dict)

        if step % FLAGS.save_model_step == 0:
            save(saver, sess, model_path, step)

        if step % FLAGS.summary_step == 0:
            gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, discriminator_sum],
                                                                   feed_dict=feed_dict)

            summary_writer.add_summary(gen_loss_sum_value, step)
            summary_writer.add_summary(discriminator_sum_value, step)

        if step % FLAGS.outfile_step == 0:
            realx_fakey, fakey_fakex, realy_fakex, fakex_fakey=sess.run([G_realx_fakey, G_fakey_fakex, G_realy_fakex, G_fakex_fakey],
                                                                        feed_dict=feed_dict)

            write_image = get_write_picture(x_image_resize, y_image_resize, realx_fakey, fakey_fakex, realy_fakex, fakex_fakey)
            write_image_name = os.path.join(outfile_path, "step-{}.png".format(str(step)))
            cv2.imwrite(write_image_name, write_image)

        if step % FLAGS.loss_step == 0:
            print('step {:d} \t all_G_loss = {:.3f}, all_D_loss = {:.3f}'.format(step, gen_loss_value, dis_loss_value))

if __name__ == '__main__':
    main()