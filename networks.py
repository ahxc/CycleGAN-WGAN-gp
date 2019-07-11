import numpy as np
import tensorflow as tf
#-----------------------------------------------------------------------------------
# 参数创建
def make_var(scope, shape, trainable=True):
    return tf.get_variable(name=scope,
                           shape=shape,
                           trainable=trainable)
#-----------------------------------------------------------------------------------
# 卷积层
def conv(X, fmaps, kernel_size, stride, padding='SAME', scope='conv', biased=False):
    channel = X.get_shape()[-1]
    with tf.variable_scope(scope):
        kernel = make_var(scope='weights',
                          shape=[kernel_size, kernel_size, channel, fmaps])

        X = tf.nn.conv2d(input=X,
                         filter=kernel,
                         stride=[1, stride, stride, 1],
                         padding=padding)

        if biased:
            biases = make_var(scope='biases',
                              shape=[fmaps])

            X = tf.nn.bias_add(X, biases)
        
        return X
#-----------------------------------------------------------------------------------
# 逆卷积层
def deconv(X, fmaps, kernel_size, stride, padding='SAME', scope='deconv'):
    channel = X.get_shape()[-1]
    height = int(X.get_shape()[1])
    width = int(X.get_shape()[2])

    with tf.variable_scope(scope):
        kernel = make_var(scope='weights',
                          shape=[kernel_size, kernel_size, fmaps, channel])
        
        X = tf.nn.conv2d_transpose(value=X,
                                   fileter=kernel,
                                   output_shape=[1, height*2, input_width*2, fmaps],
                                   strides=[1, stride, stride, 1],
                                   padding=padding)

        return X
#-----------------------------------------------------------------------------------
# 单个实例归一化层
def instance_norm(X, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(inputs=X,
                                           epsilon=1e-05,
                                           center=True,
                                           scale=True,
                                           scope=scope)
#-----------------------------------------------------------------------------------
# 批量归一化层
def batch_norm(X, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(inputs=X,
                                        epsilon=1e-5,
                                        center=True,
                                        scale=True,
                                        scope=scope)
#-----------------------------------------------------------------------------------
# 激活函数
def relu(X):
    return tf.nn.relu(X)

def tanh(X):
    return tf.nn.tanh(X)

def leaky_relu(X, alpha):
    return tf.nn.leaky_relu(X, alpha)
#-----------------------------------------------------------------------------------
# 残差块
def res_block(X, fmaps, kernel_size=3, stride=1,scope='res_block'):
    with tf.variable_scope(scope)
        f_X = conv(X=X,
                   fmaps=fmaps,
                   kernel_size=kernel_size,
                   stride=stride,
                   scope=scope+'_conv1')
        f_x = instance_norm(X=f_X, scope=scope+'_instance_norm1')
        f_x = relu(f_x)

        f_x = conv(X=f_x,
                   fmaps=fmaps,
                   kernel_size=kernel_size,
                   stride=stride,
                   scope=scope+'_conv2')
        f_x = instance_norm(X=f_X, scope=scope+'_instance_norm2')

    return relu(X+f_x)
#-----------------------------------------------------------------------------------
# 生成器
def generator(X, fmaps=64, reuse=False, scope='generator'):
    channel = X.get_shape()[-1]

    with tf.variable_scope(scope):
        if reuse:
            tf.variable_scope().reuse_variables()
        else:
            assert tf.variable_scope().reuse is False

        X = conv(X=X, fmaps=fmaps, kerner_size=7, stride=1, scope=scope+'_conv1')
        X = instance_norm(X=X, scope=scope+'_instance_norm1')
        X = relu(X)

        X = conv(X=X, fmaps=fmaps*2, kerner_size=3, stride=2, scope=scope+'_conv2')
        X = instance_norm(X=X, scope=scope+'_instance_norm2')
        X = relu(X)

        X = conv(X=X, fmaps=fmaps*4, kerner_size=3, stride=2, scope=scope+'_conv3')
        X = instance_norm(X=X, scope=scope+'_instance_norm3')
        X = relu(X)

        for i in range(1, 10):
            X = res_block(X=X, fmaps=fmaps*4, scope=scope+'res_block'+str(i))

        X = deconv(X=X, fmaps=fmaps*2, kernel_size=2, stride=2, scope=scope+"_deconv1")
        X = instance_norm(X=X, scope=scope+'_instance_norm4')
        X = relu(X)

        X = deconv(X=X, fmaps=fmaps, kernel_size=2, stride=2, scope=scope+"_deconv2")
        X = instance_norm(X=X, scope=scope+'_instance_norm5')
        X = relu(X)

        X = conv(X=X, fmaps=channel, kernel_size=7, stride=1, scope=scope+'_conv4')
    
        return tanh(X)
#-----------------------------------------------------------------------------------
# 判别器
def discriminator(X, fmaps=64, reuse=False, scope='discriminator'):
    with tf.variable_scope(scope):
        if reuse:
            tf.variable_scope().reuse_variables()
        else:
            assert tf.variable_scope().reuse is False

        X = conv(X=X, fmaps=fmaps, kernel_size=4, stride=2, scope=scope+'_conv1')
        X = leaky_relu(X)

        X = conv(X=X, fmaps=fmaps*2, kernel_size=4, stride=2, scope=scope+'_conv2')
        X = instance_norm(X=X, scope='_instance_norm1')
        X = leaky_relu(X)

        X = conv(X=X, fmaps=fmaps*4, kernel_size=4, stride=2, scope=scope+'_conv3')
        X = instance_norm(X=X, scope='_instance_norm1')
        X = leaky_relu(X)

        X = conv(X=X, fmaps=fmaps*8, kernel_size=4, stride=1, scope=scope+'_conv4')
        X = instance_norm(X=X, scope='_instance_norm1')
        X = leaky_relu(X)

        X = conv(X=X, fmaps=1, kernel_size=4, stride=1, scope=scope+'_conv5')

        return X
#-----------------------------------------------------------------------------------