# coding=utf-8
"""
Some codes from https://github.com/Newmu/dcgan_code
delete full connect in G net
"""
from __future__ import division
import math
import json
import random
import pprint
# import scipy.misc
from time import gmtime, strftime
from glob import glob
import cv2
import os
import time
# from six.moves import xrange
from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import  numpy as np

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])



def total_variation_loss(x):
    """
    Total variation loss for regularization of image smoothness
    """
    loss = tf.reduce_mean(tf.abs(x[:, 1:, :, :] - x[:, :-1, :, :])) + \
           tf.reduce_mean(tf.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    return loss


def read_font_data(font, unit_scale=True):
    # print font
    path_list = glob(font)
    path_list = sorted(path_list, key=lambda x: int(x.split('/')[-1][:].split('.')[0]))
    img_list = []

    for path in path_list:
        # print path.split('/')[-1][:].split('.')[0]
        if not path.endswith('.jpg'):
            continue
        # img = scipy.misc.imread(path).astype(np.float)
        img = cv2.imread(path)
        if unit_scale:
            img = img / 127.5 - 1
            img_list.append(img)
    return np.array(img_list)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return cv2.imwrite(path, image)
    # return scipy.misc.imsave(path, image)

def inverse_transform(images):
    img = (images + 1) * 127.5
    img = np.array(img).astype(np.uint8)
    return img
    # return  (images + 1) *127.5

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)
        # return tf.concat(axis, tensors, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias  # input_ * matrix + bias

def vgg_16_128(inputs, scale, reuse=False, pooling='avg', final_endpoint='fc8',):
    """VGG-16 implementation intended for test-time use.
    It takes inputs with values in [0, 1] and preprocesses them (scaling,
    mean-centering) before feeding them to the VGG-16 network.
    Args:
      inputs: A 4-D tensor of shape [batch_size, image_size, image_size, 3]
          and dtype float32, with values in [0, 1].
      reuse: bool. Whether to reuse model parameters. Defaults to False.
      pooling: str in {'avg', 'max'}, which pooling operation to use. Defaults
          to 'avg'.
      final_endpoint: str, specifies the endpoint to construct the network up to.
          Defaults to 'fc8'.
    Returns:
      A dict mapping end-point names to their corresponding Tensor.
    Raises:
      ValueError: the final_endpoint argument is not recognized.
    """
    inputs += 1
    inputs *= 127.5
    inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pooling_fn = pooling_fns[pooling]

    with tf.variable_scope('128_vgg_16', [inputs], reuse=reuse) as sc:
        end_points = {}

        def add_and_check_is_final(layer_name, net):
            end_points['%s/%s' % (sc.name, layer_name)] = net
            return layer_name == final_endpoint

        with slim.arg_scope([slim.conv2d], trainable=False):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            if add_and_check_is_final('conv1', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool1')
            if add_and_check_is_final('pool1', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            if add_and_check_is_final('conv2', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool2')
            if add_and_check_is_final('pool2', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            if add_and_check_is_final('conv3', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool3')
            if add_and_check_is_final('pool3', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            if add_and_check_is_final('conv4', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool4')
            if add_and_check_is_final('pool4', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if add_and_check_is_final('conv5', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool5')
            if add_and_check_is_final('pool5', net): return end_points, tf.all_variables()
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            if add_and_check_is_final('fc6', net): return end_points, tf.all_variables()
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            if add_and_check_is_final('fc7', net): return end_points, tf.all_variables()
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout7')
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None,
                              scope='fc8')
            end_points[sc.name + '/predictions'] = slim.softmax(net)
            if add_and_check_is_final('fc8', net): return end_points, tf.all_variables()

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)
def vgg_16(inputs, scale, reuse=False, pooling='avg', final_endpoint='fc8',):
    """VGG-16 implementation intended for test-time use.
    It takes inputs with values in [0, 1] and preprocesses them (scaling,
    mean-centering) before feeding them to the VGG-16 network.
    Args:
      inputs: A 4-D tensor of shape [batch_size, image_size, image_size, 3]
          and dtype float32, with values in [0, 1].
      reuse: bool. Whether to reuse model parameters. Defaults to False.
      pooling: str in {'avg', 'max'}, which pooling operation to use. Defaults
          to 'avg'.
      final_endpoint: str, specifies the endpoint to construct the network up to.
          Defaults to 'fc8'.
    Returns:
      A dict mapping end-point names to their corresponding Tensor.
    Raises:
      ValueError: the final_endpoint argument is not recognized.
    """
    inputs += 1
    inputs *= 127.5
    inputs -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pooling_fn = pooling_fns[pooling]

    with tf.variable_scope('vgg_16', [inputs], reuse=reuse) as sc:
        end_points = {}

        def add_and_check_is_final(layer_name, net):
            end_points['%s/%s' % (sc.name, layer_name)] = net
            return layer_name == final_endpoint

        with slim.arg_scope([slim.conv2d], trainable=False):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            if add_and_check_is_final('conv1', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool1')
            if add_and_check_is_final('pool1', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            if add_and_check_is_final('conv2', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool2')
            if add_and_check_is_final('pool2', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            if add_and_check_is_final('conv3', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool3')
            if add_and_check_is_final('pool3', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            if add_and_check_is_final('conv4', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool4')
            if add_and_check_is_final('pool4', net): return end_points, tf.all_variables()
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            if add_and_check_is_final('conv5', net): return end_points, tf.all_variables()
            net = pooling_fn(net, [2, 2], scope='pool5')
            if add_and_check_is_final('pool5', net): return end_points, tf.all_variables()
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            if add_and_check_is_final('fc6', net): return end_points, tf.all_variables()
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            if add_and_check_is_final('fc7', net): return end_points, tf.all_variables()
            net = slim.dropout(net, 0.5, is_training=False, scope='dropout7')
            net = slim.conv2d(net, 1000, [1, 1], activation_fn=None,
                              scope='fc8')
            end_points[sc.name + '/predictions'] = slim.softmax(net)
            if add_and_check_is_final('fc8', net): return end_points, tf.all_variables()

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

def conv_out_size_name(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def residual_block(input_, output_dim, stride=1, name='conv2d'):
    with tf.variable_scope('{}_res_block'.format(name)) as scope:
        #         print scope.name
        input_dim = input_.get_shape().as_list()[-1]
        identity = input_
        # increase dim in number of filters
        name = str(scope.name).split('/')[-1]
        if stride != 1:
            input_ = lrelu(batch_norm(name='{}_1'.format(name))(
                conv2d(input_, input_dim / 2, name='g_block_1', k_h=1, k_w=1, d_h=stride,
                       d_w=stride)))  # reduce dim / 2
            input_ = lrelu(batch_norm(name='{}_2'.format(name))(
                conv2d(input_, input_dim / 2, name='g_block_2', k_h=3, k_w=3, d_h=1, d_w=1)))  # 3*3conv2d
            input_ = batch_norm(name='{}_3'.format(name))(
                conv2d(input_, output_dim, name='g_block_3', k_h=1, k_w=1, d_h=1, d_w=1))  # increase dim
            identity = batch_norm(name='{}_4'.format(name))(
                conv2d(identity, output_dim, name='g_block_4', k_h=1, k_w=1))
        # identity map for residual blocks
        else:
            input_ = lrelu(batch_norm(name='{}_5'.format(name))(
                conv2d(input_, input_dim / 4, name='g_block_1', k_h=1, k_w=1, d_h=1, d_w=1)))  # reduce /4
            input_ = lrelu(batch_norm(name='{}_6'.format(name))(
                conv2d(input_, input_dim / 4, name='g_block_2', k_h=3, k_w=3, d_h=1, d_w=1)))
            input_ = batch_norm(name='{}_7'.format(name))(
                conv2d(input_, output_dim, name='g_block_3', k_h=1, k_w=1, d_h=1, d_w=1))

        return lrelu(identity + input_)

def histeq(image, n_bins=256):
    result = np.zeros_like(image)
    for i in range(3):
        img = (image[:, :, i] + 1) * 127.5
        imhist, bins = np.histogram(img, n_bins, normed=True)
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        x = np.interp(img.flatten(), bins[:-1], cdf)
        result[:, :, i] = x.reshape((image.shape[0], image.shape[1]))
    return (result / 127.5) - 1

class DCGAN(object):
    def __init__(self, sess, learning_rate=0.00001, beta1=0.0001, target_font='', epoch=25, train_size=0,
                 source_font='', test_path ='',
                 batch_size=64, test_num=100,
                 y_dim=None, z_dim=100, gf_dim=32, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None,
                 sample_dir=None, test_sample_dir=None,
                 source_height=128, source_width=256,
                 target_height=128, target_width=256, log_dir='', content_penalty=1e-4, first_train=False,
                 vgg_checkpoint='', dcp_dir='', train_batch='',
                 tv_penalty=0.0002, adv_penalty=1e-1, mse_penalty = 0.1):

        self.vgg_checkpoint = vgg_checkpoint
        self.vgg_checkpoint2 = vgg_checkpoint
        # self.train_batch = train_batch
        self.dcp_dir = dcp_dir
        self.sess = sess
        self.epoch = epoch
        self.z_dim = z_dim
        # self.first_train = first_train
        self.train_size = train_size
        self.batch_size = batch_size
        self.content_penalty = content_penalty
        self.test_num = test_num
        self.tv_penalty = tv_penalty
        self.adv_penalty = adv_penalty
        self.mse_penalty = mse_penalty
        self.checkpoint_dir = checkpoint_dir
        self.source_font = source_font
        self.test_path = test_path
        self.source_height = source_height
        self.source_width = source_width
        self.target_font = target_font
        self.target_height = target_height
        self.target_width = target_width
        self.sample_dir = sample_dir  # sample_dir = None
        self.test_sample_dir = test_sample_dir
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim
        self.log_dir = log_dir

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.d_bn1_128 = batch_norm(name='d_bn1_128')
        self.d_bn2_128 = batch_norm(name='d_bn2_128')
        self.d_bn3_128 = batch_norm(name='d_bn3_128')
        self.d_bn4_128 = batch_norm(name='d_bn4_128')

        self.g_bn = batch_norm(name='g_bn')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')
        self.g_bn7 = batch_norm(name='g_bn7')
        self.g_bn8 = batch_norm(name='g_bn8')
        self.g_bn9 = batch_norm(name='g_bn9')
        self.g_bn10 = batch_norm(name='g_bn10')
        self.g_bn11 = batch_norm(name='g_bn11')
        self.g_bn12 = batch_norm(name='g_bn12')
        self.g_bn13 = batch_norm(name='g_bn13')
        self.g_bn14 = batch_norm(name='g_bn14')
        self.g_bn15 = batch_norm(name='g_bn15')
        self.g_bn16 = batch_norm(name='g_bn16')
        self.g_bn17 = batch_norm(name='g_bn17')
        self.g_bn18 = batch_norm(name='g_bn18')
        self.g_bn19 = batch_norm(name='g_bn19')
        self.g_bn20 = batch_norm(name='g_bn20')

        self.l_bn = batch_norm(name='l_bn')
        self.l_bn_1 = batch_norm(name='l_bn_1')
        self.l_bn_2 = batch_norm(name='l_bn_2')
        self.l_bn_3 = batch_norm(name='l_bn_3')
        self.l_bn_4 = batch_norm(name='l_bn_4')
        self.l_bn_5 = batch_norm(name='l_bn_5')
        self.l_bn_6 = batch_norm(name='l_bn_6')
        self.l_bn_7 = batch_norm(name='l_bn_7')
        self.l_bn_8 = batch_norm(name='l_bn_8')

        self.n_bn_1 = batch_norm(name='n_bn_1')
        self.n_bn_2 = batch_norm(name='n_bn_2')
        self.n_bn_3 = batch_norm(name='n_bn_3')
        self.n_bn_5 = batch_norm(name='n_bn_5')
        self.n_bn_6 = batch_norm(name='n_bn_6')
        self.n_bn_7 = batch_norm(name='n_bn_7')
        self.build_model()

    def build_model(self):
        source_dims = [self.source_height, self.source_width, self.c_dim]  # 256*256*3
        target_dims = [self.target_height, self.target_width, self.c_dim]  # 256*256*3

        # with tf.device('/gpu:0'):  # GAN
        self.source = tf.placeholder(  # 128 * 128 * 3
            tf.float32, [self.batch_size] + source_dims, name='source'
        )

        self.source_128_128 = tf.placeholder(
            tf.float32, [self.batch_size] + [128, 128, self.c_dim], name='source_128_128'
        )
        # self.source_64_64 = tf.placeholder(
        #     tf.float32, [self.batch_size] + [64, 64, self.c_dim], name='source_64_64'
        # )
        self.target = tf.placeholder(
            tf.float32, [self.batch_size] + target_dims, name='target')  # 64*128*128*3
        self.target_128_128 = tf.placeholder(
            tf.float32, [self.batch_size] + [128,128, self.c_dim], name='target_128')  # 64*128*128*3

        # self.source_test = tf.placeholder(
        #     tf.float32, [self.batch_size] + source_dims, name='source_test')  # 64 * 128 * 128 * 3

        # self.source_256_256 = tf.placeholder(
        #      tf.float32, [self.batch_size] + source_dims, name='source_128_128'
        # )
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        self.G, self.G_128 = self.generator(self.source)  # G net

        self.sampler, self.sampler_128 = self.generator(self.source, reuse=True)

        # tv loss
        self.transform_G = (self.G + 1) * 127.5
        self.transform_G_128 = (self.G_128 + 1) * 127.5
        self.tv_loss_256 = total_variation_loss(self.transform_G) * self.tv_penalty
        self.tv_loss_128 = total_variation_loss(self.transform_G_128) * self.tv_penalty

        # adv loss
        self.d_real = self.discriminator(self.target)
        self.d_real_128 = self.discriminator_128(self.target_128_128)
        self.d_fake = self.discriminator(self.G, reuse=True)
        self.d_fake_128 = self.discriminator_128(self.G_128, reuse=True)

        # with tf.device('/gpu:1'):  # VGG
        # self.d_fake = self.discriminator(self.G, reuse=True)
        # content loss
        self.content_loss_256 = np.float32(0.0)
        G_end_points, self.vgg_variable_256 = vgg_16(self.G, 256,  reuse=False, final_endpoint='conv5')
        T_end_points, self.vgg_variable_256 = vgg_16(self.target, 256, reuse=True, final_endpoint='conv5')
        conv_layer = ['vgg_16/conv1',
                      'vgg_16/conv2']  # conv1 ~ conv5, color reconstruction ~ structure reconstruction
        for name in conv_layer:
            content_loss = tf.reduce_mean(
                tf.square(T_end_points[name] - G_end_points[name]),
                [1, 2, 3]
            )
            self.content_loss_256 += tf.reduce_mean(content_loss * self.content_penalty)

        self.content_loss_128 = np.float32(0.0)
        G_end_points, self.vgg_variable_128 = vgg_16_128(self.G_128, 128, reuse=False, final_endpoint='conv5')
        T_end_points, self.vgg_variable_128 = vgg_16_128(self.target_128_128, 128,  reuse=True, final_endpoint='conv5')
        conv_layer = ['128_vgg_16/conv1',
                      '128_vgg_16/conv2']  # conv1 ~ conv5, color reconstruction ~ structure reconstruction
        for name in conv_layer:
            content_loss = tf.reduce_mean(
                tf.square(T_end_points[name] - G_end_points[name]),
                [1, 2, 3]
            )
            self.content_loss_128 += tf.reduce_mean(content_loss * self.content_penalty)

        self.adv_loss_256 = tf.reduce_mean(-self.d_fake) * self.adv_penalty
        self.adv_loss_128 = tf.reduce_mean(-self.d_fake_128) * self.adv_penalty
        self.d_loss_256 = tf.reduce_mean(self.d_fake - self.d_real)  # WGAN的loss不取log
        self.d_loss_128 = tf.reduce_mean(self.d_fake_128 - self.d_real_128)  # WGAN的loss不取log
        self.d_loss = self.d_loss_256 + self.d_loss_128

        self.mse_loss_256 = tf.reduce_mean(self.G - self.source) * self.mse_penalty
        self.mse_loss_128 = tf.reduce_mean(self.G_128 - self.source_128_128) * self.mse_penalty

        self.g_loss_256 = self.tv_loss_256  + self.content_loss_256 + self.mse_loss_256  + self.adv_loss_256    # all (vgg)
        self.g_loss_128 = self.tv_loss_128  + self.content_loss_128 + self.mse_loss_128  + self.adv_loss_128   # all (vgg)
        self.g_loss = self.g_loss_128 + self.g_loss_256

        # with tf.device('/gpu:3'):
        t_vars = tf.trainable_variables()
        self.d_vars_256 = [var for var in t_vars if 'discriminator_256' in var.name]  #
        self.d_vars_128 = [var for var in t_vars if 'discriminator_128' in var.name]  #
        self.d_vars =     [var for var in t_vars if 'discriminator' in var.name]  #
        self.g_vars = [var for var in t_vars if 'generator' in var.name]


        # for k in self.g_vars:
        #    print k
        self.clip_D_256 = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars_256]
        self.clip_D_128 = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars_128]
        self.clip_D =     [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

        # RMSPropOptimizer函数的其他参数暂时使用默认值
        self.d_optim_256 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss_256, var_list=self.d_vars_256)
        self.d_optim_128 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss_128, var_list=self.d_vars_128)
        self.d_optim =     tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim_128 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss_128, var_list=self.g_vars)
        self.g_optim_256 = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss_256, var_list=self.g_vars)
        self.g_optim =     tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        variable_list = [v for v in self.vgg_variable_256 if 'vgg' not in v.name]
        variable_list = [v for v in self.vgg_variable_128 if 'vgg' not in v.name]
        # 只save/load GAN的参数
        self.saver = tf.train.Saver(variable_list)
        print('build model success!!!')

    def test(self, t_path, save_path):
        print('start test !!!')
        # path = './datasets/night_object2_256_256/*'
        source_npy = read_font_data(t_path).reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        # target_npy = read_font_data(self.target_font).reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])

        source_test = source_npy
        print source_test.shape
        # target_test = target_npy

        variable_list = [v for v in self.vgg_variable_256 if 'vgg' in v.name]
        saver1 = tf.train.Saver(variable_list)
        saver1.restore(self.sess, self.vgg_checkpoint)
        with tf.variable_scope("", reuse=True):

            a1 = tf.get_variable(name='128_vgg_16/conv1/conv1_1/weights', shape=[3, 3, 3, 64])
            a2 = tf.get_variable(name='128_vgg_16/conv1/conv1_1/biases', shape=[64, ], dtype='float32_ref',
                                 )
            a3 = tf.get_variable(name='128_vgg_16/conv1/conv1_2/weights', shape=[3, 3, 64, 64], dtype='float32_ref',
                                )
            a4 = tf.get_variable(name='128_vgg_16/conv1/conv1_2/biases', shape=[64, ], dtype='float32_ref',
                                 )
            a5 = tf.get_variable(name='128_vgg_16/conv2/conv2_1/weights', shape=[3, 3, 64, 128], dtype='float32_ref',
                                 )
            a6 = tf.get_variable(name='128_vgg_16/conv2/conv2_1/biases', shape=[128, ], dtype='float32_ref',
                                 )
            a7 = tf.get_variable(name='128_vgg_16/conv2/conv2_2/weights', shape=[3, 3, 128, 128], dtype='float32_ref',
                                )
            a8 = tf.get_variable(name='128_vgg_16/conv2/conv2_2/biases', shape=[128, ], dtype='float32_ref',
                                )
            a9 = tf.get_variable(name='128_vgg_16/conv3/conv3_1/weights', shape=[3, 3, 128, 256], dtype='float32_ref',
                              )
            a10 = tf.get_variable(name='128_vgg_16/conv3/conv3_1/biases', shape=[256, ], dtype='float32_ref',
                                )
            a11 = tf.get_variable(name='128_vgg_16/conv3/conv3_2/weights', shape=[3, 3, 256, 256], dtype='float32_ref',
                                 )
            a12 = tf.get_variable(name='128_vgg_16/conv3/conv3_2/biases', shape=[256, ], dtype='float32_ref',
                                 )
            a13 = tf.get_variable(name='128_vgg_16/conv3/conv3_3/weights', shape=[3, 3, 256, 256], dtype='float32_ref',
                           )
            a14 = tf.get_variable(name='128_vgg_16/conv3/conv3_3/biases', shape=[256, ], dtype='float32_ref',
                                  )
            a15 = tf.get_variable(name='128_vgg_16/conv4/conv4_1/weights', shape=[3, 3, 256, 512], dtype='float32_ref',
                                )
            a16 = tf.get_variable(name='128_vgg_16/conv4/conv4_1/biases', shape=[512, ], dtype='float32_ref',
                            )
            a17 = tf.get_variable(name='128_vgg_16/conv4/conv4_2/weights', shape=[3, 3, 512, 512], dtype='float32_ref',
                                )
            a18 = tf.get_variable(name='128_vgg_16/conv4/conv4_2/biases', shape=[512, ], dtype='float32_ref',
                                  )
            a19 = tf.get_variable(name='128_vgg_16/conv4/conv4_3/weights', shape=[3, 3, 512, 512], dtype='float32_ref',
                               )
            a20 = tf.get_variable(name='128_vgg_16/conv4/conv4_3/biases', shape=[512, ], dtype='float32_ref',
                                  )
            a21 = tf.get_variable(name='128_vgg_16/conv5/conv5_1/weights', shape=[3, 3, 512, 512], dtype='float32_ref',
                                  )
            a22 = tf.get_variable(name='128_vgg_16/conv5/conv5_1/biases', shape=[512, ], dtype='float32_ref',
                                 )
            a23 = tf.get_variable(name='128_vgg_16/conv5/conv5_2/weights', shape=[3, 3, 512, 512], dtype='float32_ref',
                                )
            a24 = tf.get_variable(name='128_vgg_16/conv5/conv5_2/biases', shape=[512, ], dtype='float32_ref',
                                 )
            a25 = tf.get_variable(name='128_vgg_16/conv5/conv5_3/weights', shape=[3, 3, 512, 512], dtype='float32_ref',
                                )
            a26 = tf.get_variable(name='128_vgg_16/conv5/conv5_3/biases', shape=[512, ], dtype='float32_ref',
                               )
        saver2 = tf.train.Saver({
            'vgg_16/conv1/conv1_1/weights': a1,
            'vgg_16/conv1/conv1_1/biases': a2,
            'vgg_16/conv1/conv1_2/weights': a3,
            'vgg_16/conv1/conv1_2/biases': a4,
            'vgg_16/conv2/conv2_1/weights': a5,
            'vgg_16/conv2/conv2_1/biases': a6,
            'vgg_16/conv2/conv2_2/weights': a7,
            'vgg_16/conv2/conv2_2/biases': a8,
            'vgg_16/conv3/conv3_1/weights': a9,
            'vgg_16/conv3/conv3_1/biases': a10,
            'vgg_16/conv3/conv3_2/weights': a11,
            'vgg_16/conv3/conv3_2/biases': a12,
            'vgg_16/conv3/conv3_3/weights': a13,
            'vgg_16/conv3/conv3_3/biases': a14,
            'vgg_16/conv4/conv4_1/weights': a15,
            'vgg_16/conv4/conv4_1/biases': a16,
            'vgg_16/conv4/conv4_2/weights': a17,
            'vgg_16/conv4/conv4_2/biases': a18,
            'vgg_16/conv4/conv4_3/weights': a19,
            'vgg_16/conv4/conv4_3/biases': a20,
            'vgg_16/conv5/conv5_1/weights': a21,
            'vgg_16/conv5/conv5_1/biases': a22,
            'vgg_16/conv5/conv5_2/weights': a23,
            'vgg_16/conv5/conv5_2/biases': a24,
            'vgg_16/conv5/conv5_3/weights': a25,
            'vgg_16/conv5/conv5_3/biases': a26
        })
        saver2.restore(self.sess, self.vgg_checkpoint2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)  # checkpoint_dir
        if could_load:
            counter_D = checkpoint_counter * 5
            counter_D_128 = counter_D
            counter_G = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_size = source_test.shape[0] // self.batch_size
        print source_test.shape
        print batch_size
        iters = 0
        itest = 0
        itest_128 = 0
        import  datetime
        starttime = datetime.datetime.now()
        print('start')
        # print 'start'
        for idx in range(0, batch_size):
            batch_source = source_test[idx * self.batch_size: (idx + 1) * self.batch_size, :, :, :]
            # batch_target = target_train[idx * self.batch_size: (idx + 1) * self.batch_size, :, :, :]
            batch_target_128_128, batch_source_128_128 = [], []
            for x in batch_source:
                batch_source_128_128.append(cv2.resize(x, (128, 128)))
            batch_source_128_128 = np.array(batch_source_128_128)
            # print batch_source_128_128
            # print batch_source
            samples, samples_128 = self.sess.run([self.sampler, self.sampler_128], feed_dict={
                self.source: batch_source,
                self.source_128_128: batch_source_128_128
            })
            for img in batch_source:
                image = (img + 1) * 127.5
                image = np.array(image).astype(np.uint8)
                path = '{}/source_test_{:d}.png'.format(save_path, iters)
                cv2.imwrite(path, image)
                iters += 1

            for i in range(self.batch_size):
                image = (samples[i, :, :, :] + 1) * 127.5
                image = np.array(image).astype(np.uint8)
                path = '{}/test_256_{:d}.png'.format(save_path, itest)
                cv2.imwrite(path, image)
                itest += 1

            for i in range(self.batch_size):
                image = (samples_128[i, :, :, :] + 1) * 127.5
                image = np.array(image).astype(np.uint8)
                path = '{}/test_128_{:d}.png'.format(save_path, itest_128)
                cv2.imwrite(path, image)
                itest_128 += 1
        endtime = datetime.datetime.now()
        print ((endtime-starttime).seconds)
        print (iters)
        print ('end')
    def train(self):
        # summary, can not assign to GPU device
        self.g_loss_sum_256 = scalar_summary("g_loss", self.g_loss_256)  #
        self.g_loss_sum_128 = scalar_summary("g_loss", self.g_loss_128)  #
        self.g_loss_sum_ = scalar_summary("g_loss", self.g_loss)  #
        self.d_loss_sum_256 = scalar_summary("d_loss_256", self.d_loss_256)
        self.d_loss_sum_128 = scalar_summary("d_loss_128", self.d_loss_128)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        self.sample_sum = image_summary('samples', self.sampler)
        self.c_loss_sum_128 = scalar_summary('content_loss', self.content_loss_128)
        self.c_loss_sum_256 = scalar_summary('content_loss', self.content_loss_256)
        self.tv_loss_sum_256 = scalar_summary('tv_loss', self.tv_loss_256)
        self.tv_loss_sum_128 = scalar_summary('tv_loss', self.tv_loss_128)
        self.mse_loss_sum_256 = scalar_summary('mse_loss', self.mse_loss_256)
        self.mse_loss_sum_128 = scalar_summary('mse_loss', self.mse_loss_128)
        """Train DCGAN"""

        # test_shuffle  = np.load('/home/kdq/enhance_img/pre-trained/test_{}.npy'.format(self.train_batch))
        # train_shuffle = np.load('/home/kdq/enhance_img/pre-trained/train_{}.npy'.format(self.train_batch))

        source_npy = read_font_data(self.source_font).reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        object_npy = read_font_data(self.test_path).reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        source_npy = np.concatenate((source_npy, object_npy), 0)
        np.random.shuffle(source_npy)

        target_npy = read_font_data(self.target_font).reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])
        tmp =  target_npy.shape[0]
        target_npy = np.concatenate((target_npy, object_npy), 0)
        for i in range(tmp, source_npy.shape[0]):
            target_npy[i] = target_npy[0]

        source_train = source_npy
        target_train = target_npy

        try:
            init = tf.global_variables_initializer()
        except:
            init = tf.initialize_all_variables()
        self.sess.run(init)

        self.writer = SummaryWriter(self.log_dir, self.sess.graph)  #####

        self.g_sum_128 = self.g_loss_sum_128
        self.g_sum_ = self.g_loss_sum_
        self.g_sum_256 = self.g_loss_sum_256
        self.d_sum_256 = self.d_loss_sum_256
        self.d_sum_128 = self.d_loss_sum_128

        counter_D = 0
        counter_D_128 = 0
        counter_G = 0
        start_time = time.time()
        could_load = False


        variable_list = [v for v in self.vgg_variable_256 if 'vgg' in v.name]
        saver1 = tf.train.Saver(variable_list)
        saver1.restore(self.sess, self.vgg_checkpoint)
        with tf.variable_scope("", reuse=True):
            a1 = tf.get_variable(name='128_vgg_16/conv1/conv1_1/weights', shape=[3, 3, 3, 64],  use_resource=True)
            a2 = tf.get_variable(name='128_vgg_16/conv1/conv1_1/biases', shape=[64, ], dtype='float32_ref', use_resource=True)
            a3 = tf.get_variable(name='128_vgg_16/conv1/conv1_2/weights', shape=[3, 3, 64, 64], dtype='float32_ref', use_resource=True)
            a4 = tf.get_variable(name='128_vgg_16/conv1/conv1_2/biases', shape=[64, ], dtype='float32_ref', use_resource=True)
            a5 = tf.get_variable(name='128_vgg_16/conv2/conv2_1/weights', shape=[3, 3, 64, 128], dtype='float32_ref', use_resource=True)
            a6 = tf.get_variable(name='128_vgg_16/conv2/conv2_1/biases', shape=[128, ], dtype='float32_ref', use_resource=True)
            a7 = tf.get_variable(name='128_vgg_16/conv2/conv2_2/weights', shape=[3, 3, 128, 128], dtype='float32_ref', use_resource=True)
            a8 = tf.get_variable(name='128_vgg_16/conv2/conv2_2/biases', shape=[128, ], dtype='float32_ref', use_resource=True)
            a9 = tf.get_variable(name='128_vgg_16/conv3/conv3_1/weights', shape=[3, 3, 128, 256], dtype='float32_ref', use_resource=True)
            a10 = tf.get_variable(name='128_vgg_16/conv3/conv3_1/biases', shape=[256, ], dtype='float32_ref', use_resource=True)
            a11 = tf.get_variable(name='128_vgg_16/conv3/conv3_2/weights', shape=[3, 3, 256, 256], dtype='float32_ref', use_resource=True)
            a12 = tf.get_variable(name='128_vgg_16/conv3/conv3_2/biases', shape=[256, ], dtype='float32_ref', use_resource=True)
            a13 = tf.get_variable(name='128_vgg_16/conv3/conv3_3/weights', shape=[3, 3, 256, 256], dtype='float32_ref', use_resource=True)
            a14 = tf.get_variable(name='128_vgg_16/conv3/conv3_3/biases', shape=[256, ], dtype='float32_ref', use_resource=True)
            a15 = tf.get_variable(name='128_vgg_16/conv4/conv4_1/weights', shape=[3, 3, 256, 512], dtype='float32_ref', use_resource=True)
            a16 = tf.get_variable(name='128_vgg_16/conv4/conv4_1/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
            a17 = tf.get_variable(name='128_vgg_16/conv4/conv4_2/weights', shape=[3, 3, 512, 512], dtype='float32_ref', use_resource=True)
            a18 = tf.get_variable(name='128_vgg_16/conv4/conv4_2/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
            a19 = tf.get_variable(name='128_vgg_16/conv4/conv4_3/weights', shape=[3, 3, 512, 512], dtype='float32_ref', use_resource=True)
            a20 = tf.get_variable(name='128_vgg_16/conv4/conv4_3/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
            a21 = tf.get_variable(name='128_vgg_16/conv5/conv5_1/weights', shape=[3, 3, 512, 512], dtype='float32_ref', use_resource=True)
            a22 = tf.get_variable(name='128_vgg_16/conv5/conv5_1/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
            a23 = tf.get_variable(name='128_vgg_16/conv5/conv5_2/weights', shape=[3, 3, 512, 512], dtype='float32_ref', use_resource=True)
            a24 = tf.get_variable(name='128_vgg_16/conv5/conv5_2/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
            a25 = tf.get_variable(name='128_vgg_16/conv5/conv5_3/weights', shape=[3, 3, 512, 512], dtype='float32_ref', use_resource=True)
            a26 = tf.get_variable(name='128_vgg_16/conv5/conv5_3/biases', shape=[512, ], dtype='float32_ref', use_resource=True)
        saver2 = tf.train.Saver({
            'vgg_16/conv1/conv1_1/weights': a1,
            'vgg_16/conv1/conv1_1/biases': a2,
            'vgg_16/conv1/conv1_2/weights': a3,
            'vgg_16/conv1/conv1_2/biases': a4,
            'vgg_16/conv2/conv2_1/weights': a5,
            'vgg_16/conv2/conv2_1/biases': a6,
            'vgg_16/conv2/conv2_2/weights': a7,
            'vgg_16/conv2/conv2_2/biases': a8,
            'vgg_16/conv3/conv3_1/weights': a9,
            'vgg_16/conv3/conv3_1/biases': a10,
            'vgg_16/conv3/conv3_2/weights': a11,
            'vgg_16/conv3/conv3_2/biases': a12,
            'vgg_16/conv3/conv3_3/weights': a13,
            'vgg_16/conv3/conv3_3/biases': a14,
            'vgg_16/conv4/conv4_1/weights': a15,
            'vgg_16/conv4/conv4_1/biases': a16,
            'vgg_16/conv4/conv4_2/weights': a17,
            'vgg_16/conv4/conv4_2/biases': a18,
            'vgg_16/conv4/conv4_3/weights': a19,
            'vgg_16/conv4/conv4_3/biases': a20,
            'vgg_16/conv5/conv5_1/weights': a21,
            'vgg_16/conv5/conv5_1/biases': a22,
            'vgg_16/conv5/conv5_2/weights': a23,
            'vgg_16/conv5/conv5_2/biases': a24,
            'vgg_16/conv5/conv5_3/weights': a25,
            'vgg_16/conv5/conv5_3/biases': a26
        })
        saver2.restore(self.sess, self.vgg_checkpoint2)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)  # checkpoint_dir
        if could_load:
            counter_D = checkpoint_counter * 5
            counter_D_128 = counter_D
            counter_G = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(self.epoch):  # config.epoch  ?????
            batch_idxs = min(source_train.shape[0], self.train_size) // self.batch_size  # //表示整除取整数商

            for idx in xrange(0, batch_idxs):
                batch_source = source_train[idx * self.batch_size: (idx + 1) * self.batch_size, :, :, :]
                batch_target = target_train[idx * self.batch_size: (idx + 1) * self.batch_size, :, :, :]
                batch_target_128_128, batch_source_128_128 = [], []
                for x in batch_source:
                    batch_source_128_128.append(cv2.resize(x, (128, 128)))
                batch_source_128_128 = np.array(batch_source_128_128)
                for x in batch_target:
                    batch_target_128_128.append(cv2.resize(x, (128, 128)))
                batch_target_128_128 = np.array(batch_target_128_128)

                # Update D network
                d_list = []
                for ___ in range(5):
                    z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                    d_loss_, _, summary_str, __ = self.sess.run([self.d_loss_256, self.d_optim_256, self.d_loss_sum_256,
                                                                 self.clip_D_256],
                                                                feed_dict={self.source: batch_source,
                                                                           self.target: batch_target,
                                                                           self.z: z,
                                                                           self.source_128_128: batch_source_128_128,
                                                                           self.target_128_128: batch_target_128_128
                                                                           })
                    d_list.append(d_loss_)
                    self.writer.add_summary(summary_str, counter_D)  # 便于查看可视化结果
                    counter_D += 1

                d_list_128 = []
                for ___ in range(5):
                    z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                    d_loss_128, _, summary_str_128, __ = self.sess.run(
                        [self.d_loss_128, self.d_optim_128, self.d_loss_sum_128, self.clip_D_128],
                        feed_dict={self.source: batch_source,
                                   self.target: batch_target,
                                   self.z: z,
                                   self.source_128_128: batch_source_128_128,
                                   self.target_128_128: batch_target_128_128
                                   })
                    d_list_128.append(d_loss_128)
                    self.writer.add_summary(summary_str_128, counter_D_128)  # 便于查看可视化结果
                    counter_D_128 += 1


                # Update G network
                g_loss_, _, summary_str= self.sess.run(
                    [self.g_loss, self.g_optim, self.g_sum_128],
                    feed_dict={self.source: batch_source,
                               self.target: batch_target,
                               self.source_128_128: batch_source_128_128,
                               self.target_128_128: batch_target_128_128
                               })


                # scalar summary write G_LOSS/TV_LOSS/CONTENT_LOSS (scale by weight)
                self.writer.add_summary(summary_str, counter_G)

                counter_G += 1

                log = "Epoch: [%2d] [%4d/%4d]  g_loss: %.4f" \
                      % (epoch, idx, batch_idxs, g_loss_)
                print(log)

                # save sample images
                if np.mod(counter_G, 100) == 1:
                    samples, samples_128 = self.sess.run([self.sampler, self.sampler_128],feed_dict={
                        self.source: batch_source,
                        self.source_128_128: batch_source_128_128,
                        self.target_128_128: batch_target_128_128
                    })
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    save_images(batch_source, [manifold_h, manifold_w],
                                '{}/train_{:02d}_{:04d}_source.png'.format(self.sample_dir, epoch, idx))
                    save_images(samples, [manifold_h, manifold_w],
                                '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    save_images(samples_128, [manifold_h, manifold_w],
                                 '{}/train_{:02d}_{:04d}_128.png'.format(self.sample_dir, epoch, idx))

                if np.mod(counter_G, 500) == 2:
                    self.save(self.checkpoint_dir, counter_G)
        self.save(self.checkpoint_dir, counter_G)

    def discriminator_128(self, image, reuse=False):
        with tf.variable_scope("discriminator_128") as scope:
            if reuse:
                scope.reuse_variables()

            # 128
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv_128'))
            # print h0.get_shape()
            # 64
            h1 = lrelu(self.d_bn1_128(conv2d(h0, self.df_dim * 2, name='d_h1_conv_128')))
            # print h1.get_shape()
            # 32
            h2 = lrelu(self.d_bn2_128(conv2d(h1, self.df_dim * 4, name='d_h2_conv_128')))
            # print h2.get_shape()
            # 16
            h3 = lrelu(self.d_bn3_128(conv2d(h2, self.df_dim * 8, name='d_h3_conv_128')))
            # print h3.get_shape()

            #  * 1
            h4_tf = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin_tf_128')

        return h4_tf
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator_256") as scope:
            if reuse:
                scope.reuse_variables()

            # 128
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

            # 64
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            # 32
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            # 16
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            # 8
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim * 16, name='d_h4_conv')))
            #  * 1
            h4_tf = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h4_lin_tf')
        return h4_tf

    def generator_local(self, source, patch, reuse=False):
        with tf.variable_scope("generator_local_{:d}".format(patch)) as scope:
            if reuse:
                scope.reuse_variables()
            return h20


    def generator(self, source, reuse = False):
        with tf.variable_scope("generator") as scope:
            return tf.nn.tanh(nh4), tf.nn.tanh(nh6)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.target_font, self.batch_size,
            self.target_height, self.target_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = self.checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        #         checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


tf.reset_default_graph()

IMAGE_SIZE = 256
EPOCH = 60
version = 'lake_epoch60'

checkpoint_dir =   '../%s/checkpoint' % version
sample_dir =       '../%s/sample' % version
log_dir =          '../%s/res' % version
test_sample_dir1 = '../%s/test/night' % version

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(test_sample_dir1):
    os.makedirs(test_sample_dir1)

run_config = tf.ConfigProto(allow_soft_placement=True)
run_config.gpu_options.allow_growth = True

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        epoch=EPOCH,
        learning_rate=0.0001,
        beta1=0.5,
        train_size=np.inf,
        batch_size=4,
        test_num=128,
        c_dim=3,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,  # "Directory name to save the checkpoints [checkpoint]"
        sample_dir=sample_dir,  # "Directory name to save the image samples [samples]"
        test_sample_dir=test_sample_dir1,
        source_font='/home/lthpc/PythonProject/N2D-GAN/datasets/lake/night_256_256/*',  # The source font npy file
        # test_path = '/home/lthpc/PythonProject/enhance_lab/datasets/object_256*256/*',
        source_height=IMAGE_SIZE,  # "The size of image to use (will be center cropped).
        source_width=IMAGE_SIZE,
        target_font='/home/lthpc/PythonProject/N2D-GAN/datasets/lake/day_256_256/*',  # "The target font npy file"
        target_height=IMAGE_SIZE,  # The size of image to use (will be center cropped)
        target_width=IMAGE_SIZE,
        tv_penalty=1e-3,
        adv_penalty=1e-1,
        mse_penalty= 1e-1,
        content_penalty=1e-5,
        vgg_checkpoint ='./vgg_16.ckpt'
        # dcp_dir='./datasets/JietiJiaoshi/night_object2_256_256/*'
    )
    dcgan.train()

