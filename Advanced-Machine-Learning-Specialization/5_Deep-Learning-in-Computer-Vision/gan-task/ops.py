# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/ops.py
#   + License: MIT

import math
import numpy as np
import tensorflow as tf
from utils import *

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,\
                                            decay=self.momentum,\
                                            updates_collections=None,\
                                            epsilon=self.epsilon,\
                                            center=True,\
                                            scale=True,\
                                            is_training=train,\
                                            scope=self.name)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    '''
    convolution the input_ tensor to new tensor with output_dim shape
    
    Parameters
    ----------
    input_: tensor shape(batch_size, height, width, dense_size)
    
    output_dim: the out_channels
    
    k_h, k_h: kernel height and width
    
    d_h, d_w: stride in height and width direction
    
    stddev: the std of the initialized weights
    
    name: the namespace of this operation-conv2d
    '''
    with tf.variable_scope(name):
        w = tf.get_variable('w',\
                            [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # conv2d input shape: [batch, in_height, in_width, in_channels]
        # conv2d kernel shape: [filter_height, filter_width, out_channels, in_channels]
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.math.abs(x)
    
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    '''
    deconvolution the input tensor to specific shape output tensor
    
    Parameters
    ----------
    input_: tensor shape(batch_size, height, width, dense_size)
    
    output_shape: shape list (batch_size. height, width, new_dense_size)
    
    k_h, k_h: kernel height and width
    
    d_h, d_w: stride in height and width direction
    
    stddev: the std of the initialized weights
    
    name: the namespace of this operation-deconv2d
    
    with_w: boolean, whether return the weight matrix
    
    Returns
    -------
    
    deconv: tensor with output shape
    '''
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels/dense_size]
        # output_shape[-1] : the number of filters
        # input_.get_shape()[-1] : the dense_size of input tensor
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            # conv2d_transpose input shape: [batch, in_height, in_width, in_channels]
            # conv2d_transpose kernel shape: [filter_height, filter_width, out_channels, in_channels]
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

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    '''
    input the noise tensor to linear operation then output tensor
    
    Parameters
    ----------
    
    input_: tensor
    
    output_size: int
    
    scope: the namespace of this operation-linear
    
    stddev: the std of the initialized weights
    
    bias_start: parameter for constant_initializer to initial
    
    with_w: boolean, whether return the weight matrix
    '''
    # create a forwar linear layer and culculate and output the result
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        # input shape (shape[0], shape[1])
        # weight matrix with shape (input_shape[1], output_shape)
        # output shape (shape[0], output_shape)
        # shape[0] is batch size
        matrix = tf.get_variable("Matrix",\
                                 [shape[1], output_size],\
                                 tf.float32,\
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
