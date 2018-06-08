# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import resnet_v2
import os, sys

def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)


def AtrousSpatialPyramidPoolingModule(inputs, depth=128, rates=[2,4,8]):
    """

    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper

    """

    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    image_features = tf.layers.conv2d(image_features, depth, [1,1], padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

    atrous_pool_block_1 = tf.layers.conv2d(inputs, depth, [1,1], padding='same', dilation_rate=1,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    atrous_pool_block_6 = tf.layers.conv2d(inputs, depth, [3,3], padding='same', dilation_rate=rates[0],
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)
    atrous_pool_block_12 = tf.layers.conv2d(inputs, depth, [3,3], padding='same', dilation_rate=rates[1],
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    atrous_pool_block_18 = tf.layers.conv2d(inputs, depth, [3,3], padding='same', dilation_rate=rates[2],
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, 
                     atrous_pool_block_12, atrous_pool_block_18), axis=3)

    return net




def build_deeplabv3_plus(inputs_, num_classes, preset_model='DeepLabV3+-Res50', weight_decay=1e-5, is_training=True,
                         drop_rate=0.5, pretrained_dir="models"):
    """
    Builds the DeepLabV3 model. 

    Arguments:
      inputs: The input tensor= 
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      DeepLabV3 model
    """
    inputs = mean_image_subtraction(inputs_)
    
    if preset_model == 'DeepLabV3_plus-Res50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            resnet_scope='resnet_v2_50'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'))
    
    elif preset_model == 'DeepLabV3_plus-Res101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            resnet_scope='resnet_v2_101'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'))
    
    elif preset_model == 'DeepLabV3_plus-Res152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            resnet_scope='resnet_v2_152'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152'))
    
    else:
        raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 50, ResNet 101, and ResNet 152" % (preset_model))


    label_size = tf.shape(inputs)[1:3]

    encoder_features = end_points['pool2']
    
    midcoder_features = AtrousSpatialPyramidPoolingModule(end_points['pool3'], rates=[4,8,12])
    midcoder_features = tf.layers.conv2d(midcoder_features, 128, [1,1], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    midcoder_features = tf.layers.batch_normalization(midcoder_features, training=is_training)
    midcoder_features = tf.maximum(0.2 * midcoder_features, midcoder_features)
    midcoder_features = tf.layers.dropout(midcoder_features, rate=drop_rate, training=is_training)
    midcoder_features = Upsampling(midcoder_features, label_size//4)
    
    decoder_features = AtrousSpatialPyramidPoolingModule(end_points['pool4'])

    decoder_features = tf.layers.conv2d(decoder_features, 128, [1,1], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    decoder_features = tf.layers.batch_normalization(decoder_features, training=is_training)
    decoder_features = tf.maximum(0.2 * decoder_features, decoder_features)
    decoder_features = tf.layers.dropout(decoder_features, rate=drop_rate, training=is_training)
    decoder_features = Upsampling(decoder_features, label_size//4)
    
    
    encoder_features = tf.layers.conv2d(encoder_features, 256, [1,1], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    encoder_features = tf.layers.batch_normalization(encoder_features, training=is_training)
    encoder_features = tf.maximum(0.2 * encoder_features, encoder_features)
    encoder_features = tf.layers.dropout(encoder_features, rate=drop_rate, training=is_training)

    net = tf.concat((encoder_features, midcoder_features, decoder_features), axis=3)
    
    net = tf.layers.conv2d(net, 256, [3,3], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.maximum(0.2 * net, net)
    net = tf.layers.dropout(net, rate=drop_rate, training=is_training)
    net = tf.layers.conv2d(net, 256, [3,3], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.maximum(0.2 * net, net)
    net = tf.layers.dropout(net, rate=drop_rate, training=is_training)

    
    net = Upsampling(net, label_size)
    
    net = tf.layers.conv2d(net, 64, [3,3], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.maximum(0.2 * net, net)
    net = tf.layers.dropout(net, rate=drop_rate, training=is_training)

    out = tf.layers.conv2d(net, num_classes, [1,1], padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),
                           kernel_regularizer=None)
    return out, init_fn


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)