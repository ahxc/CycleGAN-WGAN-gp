from networks import discriminator

import tensorflow as tf
#-----------------------------------------------------------------------------------
# 语义损失
def l1_loss(A, B):
    return tf.reduce_mean(abs(A-B))
#-----------------------------------------------------------------------------------
# 生成器损失
def g_loss(fake, GAN_type):
    if GAN_type == 'GAN':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=tf.ones_like(fake), logits=fake))

    if GAN_type == 'WGAN':
        loss = -tf.reduce_mean(fake)

    if GAN_type == 'LSGAN':
        loss = tf.reduce_mean((fake-tf.ones_like(fake)**2))

    return loss
#-----------------------------------------------------------------------------------
# 判别器损失
def d_loss(real, fake, GAN_type):
    if GAN_type == 'GAN': 
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   labels=tf.zeros_like(fake), logits=fake))

    if GAN_type == 'WGAN':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if GAN_type == 'LSGAN':
        real_loss = tf.reduce_mean((real-tf.ones_like(real))**2)
        fake_loss = tf.reduce_mean(fake**2)

    return real_loss+fake_loss
#-----------------------------------------------------------------------------------
# 梯度惩罚
def gradient_penalty(real_img, fake_image, scope='discriminator'):
    difference = real_img-fake_image
    alpha = tf.random_uniform(shape=[1,1,1,1], minval=0., maxval=1.)
    interpolates = real_img+alpha*difference

    gradients = tf.gradients(discriminator(interpolates, reuse=True, scope=scope), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1,2,3]))

    return tf.reduce_mean((slopes-1.)**2)
#-----------------------------------------------------------------------------------