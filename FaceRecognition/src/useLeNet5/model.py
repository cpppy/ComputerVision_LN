import os


import tensorflow as tf


def conv_1(data):
    kernel = tf.get_variable("conv",
                             [5, 5, 1, 32],
                             initializer=tf.random_normal_initializer())

    bias = tf.get_variable('bias',
                           [32],
                           initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data,
                        kernel,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    linear_output = tf.nn.relu(tf.add(conv, bias))

    pooling = tf.nn.max_pool(linear_output,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
    return pooling


def conv_2(data):
    kernel = tf.get_variable("conv",
                             [5, 5, 32, 64],
                             initializer=tf.random_normal_initializer())

    bias = tf.get_variable('bias',
                           [64],
                           initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data,
                        kernel,
                        strides=
                        [1, 1, 1, 1],
                        padding='SAME')

    linear_output = tf.nn.relu(tf.add(conv, bias))

    pooling = tf.nn.max_pool(linear_output,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME")
    return pooling




def fullConn_1(data):
    weights = tf.get_variable("weigths",
                              [15 * 12 * 64, 1024],
                              initializer=tf.random_normal_initializer())

    biases = tf.get_variable("biases",
                             [1024],
                             initializer=tf.random_normal_initializer())

    return tf.add(tf.matmul(data, weights), biases)


def out_layer(data):
    n_ouput_layer = 40    # predict for 40 people

    weights = tf.get_variable("weigths",
                              [1024, n_ouput_layer],
                              initializer=tf.random_normal_initializer())


    biases = tf.get_variable("biases",
                             [n_ouput_layer],
                             initializer=tf.random_normal_initializer())

    outRes = tf.add(tf.matmul(data, weights), biases)
    return outRes



def defineModel(data):
    # 根据类别个数定义最后输出层的神经元


    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("conv_layer1") as layer1:
        layer1_output = conv_1(data)

    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("conv_layer2") as layer2:
        layer2_output = conv_2(layer1_output)

    with tf.variable_scope("full_connection") as full_layer3:
        # 讲卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        layer3_output = tf.nn.relu(fullConn_1(layer2_output_flatten))

        # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    with tf.variable_scope("output") as output_layer4:
        output = out_layer(layer3_output)


    return output

