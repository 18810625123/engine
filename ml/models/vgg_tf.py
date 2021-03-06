import tensorflow as tf
import numpy as np

data_dict = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()

print(data_dict)
exit()
def print_layer(t):
    print(t.op.name, '  ', t.get_shape().as_list(), '\n')


"""
权重初始化定义了3种方式：
    1.预训练模型参数
    2.截尾正态
    3.xavier
通过参数finetrun和xavier控制选择哪种方式
"""


def conv(x, out_channel, name, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetune")
        else:
            weight = tf.Variable(tf.truncated_normal([3, 3, in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
            print("truncated normal")

        conv = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation


def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    print(activation)
    return activation


def fc(x, out_channel, name, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("finetune")
        else:
            weight = tf.Variable(tf.truncated_normal([in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
            print("truncated normal")

        # activation = tf.nn.relu_layer(x, weight, bias, name=name)
        # print_layer(activation)
        # return activation
        net = tf.add(tf.matmul(x, weight), bias)
        return net


def VGG16(images, _dropout, n_classes):
    # conv1
    conv1_1 = conv(images, 64, 'conv1_1', finetune=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', finetune=True)
    pool1 = maxpool(conv1_2, 'pool1')

    # conv2
    conv2_1 = conv(pool1, 128, 'conv2_1', finetune=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', finetune=True)
    pool2 = maxpool(conv2_2, 'pool2')

    # conv3
    conv3_1 = conv(pool2, 256, 'conv3_1', finetune=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', finetune=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', finetune=True)
    pool3 = maxpool(conv3_3, 'pool3')

    # conv4
    conv4_1 = conv(pool3, 512, 'conv4_1', finetune=True)
    conv4_2 = conv(conv4_1, 512, 'conv4_2', finetune=True)
    conv4_3 = conv(conv4_2, 512, 'conv4_3', finetune=True)
    pool4 = maxpool(conv4_3, 'pool4')

    # conv5
    conv5_1 = conv(pool4, 512, 'conv5_1', finetune=True)
    conv5_2 = conv(conv5_1, 512, 'conv5_2', finetune=True)
    conv5_3 = conv(conv5_2, 512, 'conv5_3', finetune=True)
    pool5 = maxpool(conv5_3, 'pool5')

    # fully connected layer
    flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc_6 = fc(flatten, 4096, 'fc_6', finetune=False)
    fc_6 = tf.nn.relu(fc_6)
    dropout1 = tf.nn.dropout(fc_6, _dropout)

    fc_7 = fc(dropout1, 4096, 'fc_7', finetune=False)
    fc_7 = tf.nn.relu(fc_7)
    dropout2 = tf.nn.dropout(fc_7, _dropout)

    fc_8 = fc(dropout2, n_classes, 'fc_8', finetune=False)
    return fc_8


from datetime import datetime
from VGG16 import *
import matplotlib.pyplot as plt

batch_size = 32
lr = 0.00001
n_classes = 17
max_steps = 500


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def train():
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(tf.int64, shape=[None, n_classes], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, n_classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    images, labels = read_and_decode('train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=512,
                                                    min_after_dequeue=200)
    label_batch = tf.one_hot(label_batch, n_classes, 1, 0)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    plot_loss = []
    fig = plt.figure()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps):
            batch_x, batch_y = sess.run([img_batch, label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            if i % 100 == 0:
                train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                print("%s: Step [%d] Loss: %f, training accuracy: %g" % (datetime.now(), i, loss_val, train_acc))

            plot_loss.append(loss_val)

            if (i + 1) == max_steps:
                saver.save(sess, './model/model.ckpt', global_step=i)

        coord.request_stop()
        coord.join(threads)

train()