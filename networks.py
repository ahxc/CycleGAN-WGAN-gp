# author: zhangodie
#-----------------------------------------------------------------------------------
# 网络基础架构

# 输入图像的张量形状
# [batch, height, width, channel]

# 构造可训练参数(核，偏置等)
#构造可训练参数

import numpy as np
import tensorflow as tf

def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)
"""
get_variable(
    name, 名字
    shape=None, 形状
    dtype=None, 类型
    initializer=None, 初始化
    regularizer=None,
    trainable=True, 为True加入图集
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)
"""
 
# 定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]# 获取通道数
    with tf.variable_scope(name):# 命名作用域，通常与get_variable配合使用，实现变量共享
        
        # size * size，通道数，核个数(决定了输出个数)
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])

        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)# 加上偏差项
        
        return output
 
# 定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    
    input_dim = input_.get_shape()[-1]
    
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])

        # 输入，核，空洞填充，padding
        # dilation = 1则与普通卷积无异，为2隔一个采样点
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        
        return output

# 定义反卷积层
# 反卷积类似于空洞卷积的逆过程，空洞卷积隔(忽略)一个像素进行采样，反卷积填充一个像素进行采样
# 反卷积不能称为真正意义上的反卷积，更准确的说为转置卷积，应为他无法还原值，只能还原尺寸
def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])# 获得图像高
    input_width = int(input_.get_shape()[2])# 获得图像宽

    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        
        # 输入，核，输出形状(放大两倍)与个数，步长，padding方式
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = padding)
        return output
 
# 定义batchnorm(批量归一化)层
def batch_norm(input_, name = "batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]

        # 均值为1，标准差为0.02的初始化
        scale = tf.get_variable("scale", [input_dim], initializer = tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))

        # 一个标量，形状与值会自适应，值以最后一个元素广播
        offset = tf.get_variable("offset", [input_dim], initializer = tf.constant_initializer(0.0))

        # 求输入1，2列的均值和方差
        mean, variance = tf.nn.moments(input_, axes = [1,2], keep_dims = True)
        epsilon = 1e-5# 1 乘以 10 的 -5 次方
        inv = tf.rsqrt(variance + epsilon)# 平方根的倒数
        normalized = (input_ - mean) * inv# 标准化

        return scale * normalized + offset# 线性函数

# 定义实例归一化层
def instance_norm(input_, scope = 'instance_norm'):
    return tf.contrib.layers.instance_norm(input_, epsilon = 1e-05, center = True, scale = True, scope = scope)
 
#定义最大池化层
def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize = [1, kernel_size, kernel_size, 1], strides = [1, stride, stride, 1], padding = padding, name = name)
 
#定义平均池化层
def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize = [1, kernel_size, kernel_size, 1], strides = [1, stride, stride, 1], padding = padding, name = name)
 
#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak * x)
 
#定义relu激活层
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)

# 定义tanh激活层
def tanh(input_, name = "tanh"):
    return tf.nn.tanh(input_, name = name)

# 定义残差块
def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "resnet"):
    if atrous:# 采用空洞卷积的方式
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_conv1'))
        conv2dc0_norm = instance_norm(input_ = conv2dc0, scope = (name + '_insn_1'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_conv2'))
        conv2dc1_norm = instance_norm(input_ = conv2dc1, scope = (name + '_insn_2'))
    else:# 采用卷积的方式
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_conv1'))
        conv2dc0_norm = instance_norm(input_ = conv2dc0, scope = (name + '_insn_1'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_conv2'))
        conv2dc1_norm = instance_norm(input_ = conv2dc1, scope = (name + '_insn_2'))

    return relu(input_ = input_ + conv2dc1_norm)

'''
# wgan-gp损失定义方法
# 生成器G：
G_loss = -tf.reduce_mean(D(G(real)))

# 判别器D：
D_real = D(real)
fake = G(real)
D_fake = D(fake)
D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
# GP梯度惩罚
differences = fake - real
alpha = tf.random_uniform(shape = [BATCH_SIZE, 1, 1, 1], minval = 0., maxval = 1.)# tf.random_uniform从均匀分布中生成形状为shape，满足最大最小的随机数
interpolates = real + alpha * differences
gradients = tf.gradients(D(interpolates), [interpolates])[0]# 参数y，x，返回每个x的sum(dy/dx)取第一列
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1,2,3]))# reduction_indices：保留的维度
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
# 加上梯度惩罚
D_loss += LAMBDA * gradient_penalty# lambda梯度惩罚超参数，默认为10
# pggan复现作者添加了额外一项
D_loss += tf.reduce_mean(tf.square(D_real))* 0.001
'''

# 定义l1_loss，A域原图像与生成A域原图像之间的损失
# 只用到重建生成器的训练
def l1_loss(src, dst):
    return tf.reduce_mean(tf.abs(src - dst))

# 定义G的loss
def G_loss(D_fake):
    return tf.reduce_mean(D_fake)

# 定义D的loss
def D_loss(D_real, D_fake):
    return tf.reduce_mean(D_real ** 2) + tf.reduce_mean(D_fake ** 2)

# 定义判别器损失的梯度惩罚
def gradient_penalty(real, fake, gp_lambda, reuse = False, name = 'discriminator'):
    difference = fake - real# 生成图像与真实图像的差异

    # tf.random_uniform从均匀分布中生成形状为shape，满足最大最小的随机数
    alpha = tf.random_uniform(shape = [1, 1, 1, 1], minval = 0., maxval = 1.)
    interpolates = real + alpha * difference# 真实图像加上差异

    # 求导返回值是一个list，list的长度等于x个数
    # 参数y，x，xy为多值或单一值，返回值：[sum(dy_1/dx_1, ... ,dy_n/dx_1), sum(dy_1/dx_2, ... ,dy_n/dx_2)]
    gradients = tf.gradients(discriminator(interpolates, reuse = reuse, name = name), [interpolates])[0]# 返回值是包含一个元素的list

    # reduction_indices：废弃的维度，直观理解就是将指定维度加到剩余维度上变成剩余维度
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices = [1,2,3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return gp_lambda * gradient_penalty

# 定义生成器
def generator(image, gf_dim=64, reuse=False, name="generator"):
    # 生成器输入尺度: 1*256*256*3 一张256*256的3通道图
    input_dim = image.get_shape()[-1]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()# 设置变量为共享
        else:
            assert tf.get_variable_scope().reuse is False# 如果assert后的语句为False，则报错
        # 第1个卷积模块，输出尺度: 1*256*256*64  
        c0 = relu(instance_norm(conv2d(input_ = image, output_dim = gf_dim, kernel_size = 7, stride = 1, name = 'g_e0_c'), scope = 'g_insn_1'))
        # 第2个卷积模块，输出尺度: 1*128*128*128
        c1 = relu(instance_norm(conv2d(input_ = c0, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_e1_c'), scope = 'g_insn_2'))
        # 第3个卷积模块，输出尺度: 1*64*64*256
        c2 = relu(instance_norm(conv2d(input_ = c1, output_dim = gf_dim * 4, kernel_size = 3, stride = 2, name = 'g_e2_c'), scope = 'g_insn_3'))

        # 9个残差块:
        r1 = residule_block_33(input_ = c2, output_dim = gf_dim * 4, atrous = False, name = 'g_r1')
        r2 = residule_block_33(input_ = r1, output_dim = gf_dim * 4, atrous = False, name = 'g_r2')
        r3 = residule_block_33(input_ = r2, output_dim = gf_dim * 4, atrous = False, name = 'g_r3')
        r4 = residule_block_33(input_ = r3, output_dim = gf_dim * 4, atrous = False, name = 'g_r4')
        r5 = residule_block_33(input_ = r4, output_dim = gf_dim * 4, atrous = False, name = 'g_r5')
        r6 = residule_block_33(input_ = r5, output_dim = gf_dim * 4, atrous = False, name = 'g_r6')
        r7 = residule_block_33(input_ = r6, output_dim = gf_dim * 4, atrous = False, name = 'g_r7')
        r8 = residule_block_33(input_ = r7, output_dim = gf_dim * 4, atrous = False, name = 'g_r8')
        r9 = residule_block_33(input_ = r8, output_dim = gf_dim * 4, atrous = False, name = 'g_r9')
        # 第9个残差块的输出尺度: 1*64*64*256
 
        # 第1个反卷积模块，输出尺度: 1*128*128*128
        d1 = relu(instance_norm(deconv2d(input_ = r9, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_d1_dc'), scope = 'g_insn_4'))
        # 第2个反卷积模块，输出尺度: 1*256*256*64
        d2 = relu(instance_norm(deconv2d(input_ = d1, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d2_dc'), scope = 'g_insn_5'))
        # 最后一个卷积模块，输出尺度: 1*256*256*3(还原至输入情况)
        d3 = conv2d(input_=d2, output_dim  = input_dim, kernel_size = 7, stride = 1, name = 'g_d3_c')
        # 经过tanh函数激活得到生成的输出
        return tf.nn.tanh(d3)

# 定义判别器
def discriminator(image, df_dim=64, reuse = False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # 第1个卷积模块，输出尺度: 1*128*128*64
        h0 = lrelu(conv2d(input_ = image, output_dim = df_dim, kernel_size = 4, stride = 2, name = 'd_h0_conv'))
        # 第2个卷积模块，输出尺度: 1*64*64*128
        h1 = lrelu(instance_norm(conv2d(input_ = h0, output_dim = df_dim * 2, kernel_size = 4, stride = 2, name = 'd_h1_conv'), scope = 'd_insn_1'))
        # 第3个卷积模块，输出尺度: 1*32*32*256
        h2 = lrelu(instance_norm(conv2d(input_ = h1, output_dim = df_dim * 4, kernel_size = 4, stride = 2, name = 'd_h2_conv'), scope = 'd_insn_2'))
        # 第4个卷积模块，输出尺度: 1*32*32*512
        h3 = lrelu(instance_norm(conv2d(input_ = h2, output_dim = df_dim * 8, kernel_size = 4, stride = 1, name = 'd_h3_conv'), scope = 'd_insn_3'))
        # 最后一个卷积模块，输出尺度: 1*32*32*1
        output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, name = 'd_h4_conv')
        return output