"""
Networks for GAN Pix2Pix.

1. Instance Normalization in G only (uncomment 251 and comment 252 to use IN in G and D).
2. Hinge loss.
3. Spectral Normalization in D only.

4. D's output is [N, 30, 30, 1], 6 layers
"""

# import os
# import sys
#
# sys.path.append(os.getcwd())

import numpy as np
# import common as lib
import common.ops.conv2d
import common.ops.linear
import common.ops.normalization

from common.resnet_block import *


def norm_layer(inputs, decay=0.9, epsilon=1e-5, is_training=True, norm_type="BN"):
    """

    Args:
      inputs: A batch of images to be normed. Shape is [batch, height, width, channels].
      decay: Default value: 1e-5 for BatchNorm, 1e-6 for InstanceNorm.
      epsilon: Default value: 1e-5 for BatchNorm, 1e-6 for InstanceNorm.
      is_training: .
      norm_type: "BN" for BatchNorm, "IN" for InstanceNorm.

    Returns:
      Returns normalized image batch.
    """
    if norm_type == "BN":
        outputs = lib.ops.normalization.batch_norm(inputs, decay=decay, epsilon=epsilon, is_training=True)
    elif norm_type == "IN":
        outputs = lib.ops.normalization.instance_norm(inputs, epsilon=epsilon, trainable=is_training)
    else:
        raise NotImplementedError('Normalization [%s] is not implemented!' % norm_type)

    return outputs


def Self_Atten(x, spectral_normed=True):
    """

    Args:
      x: [b_size, f_size, f_size, in_dim]
      pixel_wise:
    Return:
      [batch_size, H, W, in_dim]
    """
    with tf.variable_scope('Self_Attn'):
        N = x.shape.as_list()[0]
        H = x.shape.as_list()[1]
        W = x.shape.as_list()[2]
        in_dim = x.shape.as_list()[-1]

        gamma = tf.get_variable(name='gamma', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

        # [N, H, W, in_dim // 8]
        f = \
            lib.ops.conv2d.Conv2D(x, x.shape.as_list()[-1], in_dim // 8, filter_size=1, stride=1,
                                  name='Conv2D.f', conv_type='conv2d', channel_multiplier=0, padding='SAME',
                                  spectral_normed=spectral_normed,
                                  update_collection=None,
                                  inputs_norm=False,
                                  he_init=True, biases=True)
        f_ready = tf.reshape(f, [N, H * W, -1])  # [N, H*W, in_dim // 8]

        # [N, H, W, in_dim // 8]
        g = \
            lib.ops.conv2d.Conv2D(x, x.shape.as_list()[-1], in_dim // 8, filter_size=1, stride=1,
                                  name='Conv2D.g', conv_type='conv2d', channel_multiplier=0, padding='SAME',
                                  spectral_normed=spectral_normed,
                                  update_collection=None,
                                  inputs_norm=False,
                                  he_init=True, biases=True)
        g_ready = tf.reshape(g, [N, H * W, -1])  # [N, H*W, in_dim // 8]

        energy = tf.matmul(g_ready, f_ready, transpose_b=True)  # [N, H*W, H*W]
        attention = tf.nn.softmax(energy, axis=-1)  # [N, H*W, H*W]

        # [N, H, W, in_dim]
        h = \
            lib.ops.conv2d.Conv2D(x, in_dim, in_dim, filter_size=1, stride=1,
                                  name='Conv2D.h', conv_type='conv2d', channel_multiplier=0, padding='SAME',
                                  spectral_normed=spectral_normed,
                                  update_collection=None,
                                  inputs_norm=False,
                                  he_init=True, biases=True)
        h = tf.reshape(h, [N, H * W, -1])  # [N, H*W, in_dim]

        out = tf.matmul(attention, h)  # [N, H*W, in_dim]
        out = tf.reshape(out, [N, H, W, in_dim])  # [N, in_dim, W, H]

        self_attn_map = gamma * out + x

        return self_attn_map, attention


# ######################## ResNet ######################## #


def resnet_g(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
             upsampe_method='depth_to_space'):
    """ UNet using ResNet architecture.
    Args:

    Returns:
    """
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 512, 512, ngf]
    with tf.variable_scope("layer_1"):
        inputs = tf.pad(generator_inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        print('resnet_g.inputs: {}'.format(inputs.shape.as_list()))

        output = lib.ops.conv2d.Conv2D(
            inputs, inputs.shape.as_list()[-1], ngf, 7, 1, 'Conv2D',
            conv_type='conv2d', channel_multiplier=0,
            padding='VALID', spectral_normed=True, update_collection=None,
            inputs_norm=False, he_init=True, biases=True)
        output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
        output = nonlinearity(output, 'lrelu', 0.2)

        layers.append(output)
        print('resnet_g: {}'.format(layers[-1].shape.as_list()))

    n_downsampling = 4  # [batch, 512, 512, ngf] ----> [batch, 32, 32, ngf * 8]
    for i in range(n_downsampling):
        with tf.variable_scope('layer_{}'.format(len(layers) + 1)):
            mult = min(4, 2 ** i) * 2  # 2, 4, 8, 8
            output = lib.ops.conv2d.Conv2D(
                layers[-1], layers[-1].shape.as_list()[-1], ngf * mult, 3, 2, 'Conv2D',
                conv_type='conv2d', channel_multiplier=0,
                padding='SAME', spectral_normed=True, update_collection=None,
                inputs_norm=False, he_init=True, biases=True)
            output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            output = nonlinearity(output, 'lrelu', 0.2)

            layers.append(output)
            print('resnet_g: {}'.format(layers[-1].shape.as_list()))

    # [batch, 32, 32, ngf * 8] ----> [batch, 32, 32, ngf * 8]
    mult = min(8, 2 ** n_downsampling)
    n_block = 6
    for i in range(n_block):
        with tf.variable_scope('layer_{}'.format(len(layers) + 1)):
            output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ngf * mult, 3,
                                   name='G.Block.%d' % (len(layers) + 1),
                                   spectral_normed=True,
                                   update_collection=None,
                                   inputs_norm=False,
                                   resample=None, labels=None, biases=True, activation_fn='relu')

            layers.append(output)
            print('resnet_g: {}'.format(layers[-1].shape.as_list()))

    # [batch, 32, 32, ngf * 8] ----> [batch, 512, 512, ngf]
    for i in range(n_downsampling):
        with tf.variable_scope('layer_{}'.format(len(layers) + 1)):
            mult = min(8, 2 ** (n_downsampling - i - 1))

            inputs = tf.concat([layers[-1], layers[-1], layers[-1], layers[-1]], axis=3)
            inputs = tf.depth_to_space(inputs, 2)
            output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], ngf * mult, 3, 1,
                                           name='Conv2D', conv_type='conv2d', channel_multiplier=0, padding='SAME',
                                           spectral_normed=True, update_collection=None, inputs_norm=False,
                                           he_init=True, biases=True)
            output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            output = nonlinearity(output, 'relu', 0.2)

            if i in [1]:
                output, attn_score = Self_Atten(output)  # attention module
                print('Self_Atten.D: {}'.format(output.shape.as_list()))

            layers.append(output)
            print('resnet_g: {}'.format(layers[-1].shape.as_list()))

    # [batch, 512, 512, ngf] ----> [batch, 512, 512, 3]
    with tf.variable_scope('layer_{}'.format(len(layers) + 1)):
        inputs = tf.pad(layers[-1], [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], generator_outputs_channels, 7, 1,
                                       name='Conv2D', conv_type='conv2d', channel_multiplier=0, padding='VALID',
                                       spectral_normed=True, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        # output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
        output = tf.nn.tanh(output)

        layers.append(output)
        print('resnet_g: {}'.format(layers[-1].shape.as_list()))

    return layers[-1]


def resnet_g_1(generator_inputs, generator_outputs_channels, ngf):
    """ UNet using ResNet architecture.
    Args:

    Returns:
    """
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = ResidualBlock(
            generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 3,
            name='G.Block.1',
            spectral_normed=True, update_collection=None, inputs_norm=False,
            resample='down', labels=None, biases=True, activation_fn='relu')

        layers.append(output)
        print('G.encoder_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 16,  # encoder_6: [batch, 16, 16, ngf * 16] => [batch, 8, 8, ngf * 16]
        ngf * 16,  # encoder_7: [batch, 8, 8, ngf * 16] => [batch, 4, 4, ngf * 16]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            output = ResidualBlock(
                layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                name='G.Block.%d' % (len(layers) + 1),
                spectral_normed=True, update_collection=None, inputs_norm=False,
                resample='down', labels=None, biases=True, activation_fn='relu')

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)
            print('G.encoder_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    # [batch, 4, 4, ngf * 16] ----> [batch, 512, 512, ngf]
    layer_specs_ = [
        ngf * 16,  # encoder_7: [batch, 4, 4, ngf * 16] => [batch, 8, 8, ngf * 16]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 16] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 8]
        ngf * 4,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 64, 64, ngf * 4]
        ngf * 2,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 128, 128, ngf * 2]
        ngf * 1,  # encoder_2: [batch, 128, 128, ngf * 2] => [batch, 256, 256, ngf]
        ngf * 1,  # encoder_1: [batch, 256, 256, ngf] => [batch, 512, 512, ngf]
    ]
    for out_channels in layer_specs_:
        with tf.variable_scope('decoder_{}'.format(len(layers) - len(layer_specs))):
            output = ResidualBlock(
                layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                name='G.Block.%d' % (len(layers) - len(layer_specs)),
                spectral_normed=True, update_collection=None, inputs_norm=False,
                resample='up', labels=None, biases=True, activation_fn='relu')

            if out_channels == ngf * 4:
                output, attn_score = Self_Atten(output, spectral_normed=True)  # attention module
                print('Self_Atten.G: {}'.format(output.shape.as_list()))

            layers.append(output)
            print('G.decoder_{}: {}'.format(len(layers) - len(layer_specs) - 1, layers[-1].shape.as_list()))

    # [batch, 512, 512, ngf] ----> [batch, 512, 512, 3]
    with tf.variable_scope('decoder_{}'.format(len(layers) - len(layer_specs))):
        output = norm_layer(layers[-1], decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")
        output = nonlinearity(output)

        # output = tf.pad(output, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        output = lib.ops.conv2d.Conv2D(
            output, output.shape.as_list()[-1], generator_outputs_channels, 3, 1, 'Conv2D',
            conv_type='conv2d', channel_multiplier=0, padding='SAME',
            spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)

        output = tf.nn.tanh(output)
        layers.append(output)
        print('G.decoder_{}: {}'.format(len(layers) - len(layer_specs) - 1, layers[-1].shape.as_list()))

    return layers[-1]


def resnet_g_vgg(generator_inputs, generator_outputs_channels, ngf, vgg19_npy_path=None):
    """ Using vgg to encode image, and ResNet architecture to decode image.
    Args:

    Returns:
    """
    layers = []

    if vgg19_npy_path is not None:
        data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    else:
        data_dict = None

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("vgg"):
        rgb_scaled = (generator_inputs + 1) / 2  # [-1, 1] => [0, 1]
        rgb_scaled *= 255.0
        # Convert RGB to BGR
        red, green, blue = tf.split(value=rgb_scaled, num_or_size_splits=3, axis=3)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        conv1_1 = conv_layer(bgr, 6, 64, "conv1_1", trainable=False, data_dict=None)
        conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2", trainable=False, data_dict=data_dict)
        pool1 = max_pool(conv1_2, 'pool1')  # [112, 112, 64], [256, 256, 64]

        conv2_1 = conv_layer(pool1, 64, 128, "conv2_1", trainable=False, data_dict=data_dict)
        conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2", trainable=False, data_dict=data_dict)
        pool2 = max_pool(conv2_2, 'pool2')  # [56, 56, 128], [128, 128, 128]

        conv3_1 = conv_layer(pool2, 128, 256, "conv3_1", trainable=False, data_dict=data_dict)
        conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2", trainable=False, data_dict=data_dict)
        conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3", trainable=False, data_dict=data_dict)
        conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4", trainable=False, data_dict=data_dict)
        pool3 = max_pool(conv3_4, 'pool3')  # [28, 28, 256], [64, 64, 256]

        conv4_1 = conv_layer(pool3, 256, 512, "conv4_1", trainable=True, data_dict=data_dict)
        conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2", trainable=True, data_dict=data_dict)
        conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3", trainable=True, data_dict=data_dict)
        conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4", trainable=True, data_dict=data_dict)
        pool4 = max_pool(conv4_4, 'pool4')  # [14, 14, 512], [32, 32, 512]

        conv5_1 = conv_layer(pool4, 512, 512, "conv5_1", trainable=True, data_dict=data_dict)
        conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2", trainable=True, data_dict=data_dict)
        conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3", trainable=True, data_dict=data_dict)
        conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4", trainable=True, data_dict=data_dict)
        pool5 = max_pool(conv5_4, 'pool5')  # [7, 7, 512], [16, 16, 512]

        output = ResidualBlock(
            generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 3,
            name='G.Block.1',
            spectral_normed=True, update_collection=None, inputs_norm=False,
            resample='down', labels=None, biases=True, activation_fn='relu')

        # inputs = tf.pad(generator_inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        # print('resnet_g.inputs: {}'.format(inputs.shape.as_list()))
        # output = lib.ops.conv2d.Conv2D(
        #     inputs, inputs.shape.as_list()[-1], ngf, 5, 2, 'Conv2D',
        #     conv_type='conv2d', channel_multiplier=0, padding='VALID',
        #     spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)

        layers.append(output)
        print('G.encoder_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 16,  # encoder_6: [batch, 16, 16, ngf * 16] => [batch, 8, 8, ngf * 16]
        ngf * 16,  # encoder_7: [batch, 8, 8, ngf * 16] => [batch, 4, 4, ngf * 16]
        # ngf * 8,  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            output = ResidualBlock(
                layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                name='G.Block.%d' % (len(layers) + 1),
                spectral_normed=True, update_collection=None, inputs_norm=False,
                resample='down', labels=None, biases=True, activation_fn='relu')

            # output = norm_layer(layers[-1], decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")
            # output = nonlinearity(output, 'lrelu', 0.2)
            # inputs = tf.pad(generator_inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
            # output = lib.ops.conv2d.Conv2D(
            #     output, output.shape.as_list()[-1], out_channels, 5, 2, 'Conv2D',
            #     conv_type=conv_type, channel_multiplier=channel_multiplier,
            #     padding=padding, spectral_normed=True, update_collection=None,
            #     inputs_norm=False, he_init=True, biases=True)

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)
            print('G.encoder_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    # [batch, 4, 4, ngf * 16] ----> [batch, 512, 512, ngf]
    layer_specs_ = [
        ngf * 16,  # encoder_7: [batch, 4, 4, ngf * 16] => [batch, 8, 8, ngf * 16]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 16] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 8]
        ngf * 4,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 64, 64, ngf * 4]
        ngf * 2,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 128, 128, ngf * 2]
        ngf * 1,  # encoder_2: [batch, 128, 128, ngf * 2] => [batch, 256, 256, ngf]
        ngf * 1,  # encoder_1: [batch, 256, 256, ngf] => [batch, 512, 512, ngf]
    ]
    for out_channels in layer_specs_:
        with tf.variable_scope('decoder_{}'.format(len(layers) - len(layer_specs))):
            output = ResidualBlock(
                layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                name='G.Block.%d' % (len(layers) - len(layer_specs)),
                spectral_normed=True, update_collection=None, inputs_norm=False,
                resample='up', labels=None, biases=True, activation_fn='relu')

            if out_channels == ngf * 4:
                output, attn_score = Self_Atten(output, spectral_normed=True)  # attention module
                print('Self_Atten.G: {}'.format(output.shape.as_list()))

            layers.append(output)
            print('G.decoder_{}: {}'.format(len(layers) - len(layer_specs) - 1, layers[-1].shape.as_list()))

    # [batch, 512, 512, ngf] ----> [batch, 512, 512, 3]
    with tf.variable_scope('decoder_{}'.format(len(layers) - len(layer_specs))):
        output = norm_layer(layers[-1], decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")
        output = nonlinearity(output)

        # output = tf.pad(output, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        output = lib.ops.conv2d.Conv2D(
            output, output.shape.as_list()[-1], generator_outputs_channels, 3, 1, 'Conv2D',
            conv_type='conv2d', channel_multiplier=0, padding='SAME',
            spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)

        output = tf.nn.tanh(output)
        layers.append(output)
        print('G.decoder_{}: {}'.format(len(layers) - len(layer_specs) - 1, layers[-1].shape.as_list()))

    return layers[-1]


# TODO
def resnet_d(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
             conv_type, channel_multiplier, padding):
    """
    Args:
      discrim_inputs: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      [N, 1]
    """
    n_downsampling = 5
    layers = []

    # 2 x [batch, 512, 512, in_channels] => [batch, 512, 512, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        output = OptimizedResBlockDisc1(inputs, DIM_D=ndf, activation_fn='relu',
                                        spectral_normed=spectral_normed,
                                        update_collection=None,
                                        inputs_norm=False,
                                        biases=True)

        layers.append(output)
        print('resnet_d: {}'.format(layers[-1].shape.as_list()))

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 8] => [batch, 16, 16, ndf * 8]
    # layer_6: [batch, 16, 16, ndf * 8] => [batch, 8, 8, ndf * 8]
    for i in range(n_downsampling):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2 ** (i + 1), 8)
            output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                                   name='D.Block.%d' % (len(layers) + 1),
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=False,
                                   resample='down', labels=None, biases=True, activation_fn='relu')

            if i == 0:
                output, attn_score = Self_Atten(output)  # attention module
                print('Self_Atten.D: {}'.format(output.shape.as_list()))

            layers.append(output)
            print('resnet_d: {}'.format(layers[-1].shape.as_list()))

    # layer_7: [batch, 8, 8, ndf * 8] => [batch, 8, 8, ndf * 4]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ndf * 4, 3,
                               name='D.Block.%d' % (len(layers) + 1),
                               spectral_normed=spectral_normed,
                               update_collection=update_collection,
                               inputs_norm=False,
                               resample=None, labels=None, biases=True, activation_fn='relu')
        layers.append(output)
        print('resnet_d: {}'.format(layers[-1].shape.as_list()))

    # layer_8: [batch, 8, 8, ndf * 4] => [batch, 8, 8, ndf * 2]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ndf, 3,
                               name='D.Block.%d' % (len(layers) + 1),
                               spectral_normed=spectral_normed,
                               update_collection=update_collection,
                               inputs_norm=False,
                               resample=None, labels=None, biases=True, activation_fn='relu')
        print('resnet_d: {}'.format(layers[-1].shape.as_list()))

        output = nonlinearity(output, 'lrelu', 0.2)
        output = tf.reduce_mean(output, axis=[1, 2])
        output_wgan = lib.ops.linear.Linear(output, ndf, 1, 'D.Output',
                                            spectral_normed=spectral_normed,
                                            update_collection=update_collection)
        output_wgan = tf.reshape(output_wgan, [-1])
        layers.append(output_wgan)
        print('resnet_d: {}'.format(layers[-1].shape.as_list()))

    return layers[-1]


# TODO
def resnet_d_(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
              conv_type, channel_multiplier, padding):
    """
    Args:
      discrim_inputs: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      [N, 30, 30, ndf]
    """

    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        output = OptimizedResBlockDisc1(inputs, DIM_D=ndf, activation_fn='lrelu',
                                        spectral_normed=spectral_normed,
                                        update_collection=None,
                                        inputs_norm=False,
                                        biases=True)
        output = nonlinearity(output, 'lrelu', 0.2)
        layers.append(output)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], out_channels_, 3,
                                   name='D.Block.%d' % (len(layers) + 1),
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=False,
                                   resample='down', labels=None, biases=True, activation_fn='lrelu')

            # if i == 2:
            #     output = nonlinearity(output, 'lrelu', 0.2)
            #     output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)

    print('1.shape: {}'.format(layers[-1].shape.as_list()))

    # layer_5: [batch, 32, 32, ndf * 8] => [batch, 32, 32, ndf * 2]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ndf * 2, 3,
                               name='D.Block.%d' % (len(layers) + 1),
                               spectral_normed=spectral_normed,
                               update_collection=update_collection,
                               inputs_norm=False,
                               resample=None, labels=None, biases=True, activation_fn='lrelu')

        layers.append(output)

        print('2.shape: {}'.format(layers[-1].shape.as_list()))

    # layer_6: [batch, 32, 32, ndf * 8] => [batch, 32, 32, ndf * 2]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ndf * 2, 3,
                               name='D.Block.%d' % (len(layers) + 1),
                               spectral_normed=spectral_normed,
                               update_collection=update_collection,
                               inputs_norm=False,
                               resample=None, labels=None, biases=True, activation_fn='lrelu')

        output = nonlinearity(output, 'lrelu', 0.2)

        output, attn_score = Self_Atten(output)  # attention module

        layers.append(output)

        print('2.shape: {}'.format(layers[-1].shape.as_list()))

    # layer_7: [batch, 32, 32, ndf * 8] => [batch, 31, 31, ndf * 8]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        # output = nonlinearity(layers[-1], 'lrelu', 0.2)
        padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf * 8, 4, 1,
                                          name='Conv2D',
                                          conv_type=conv_type,
                                          channel_multiplier=channel_multiplier,
                                          padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)

        # normalized = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        rectified, attn_score = Self_Atten(rectified)  # attention module

        layers.append(rectified)

    print('2.shape: {}'.format(layers[-1].shape.as_list()))

    # layer_8: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], 1, 4, 1,
                                          name='Conv2D',
                                          conv_type=conv_type,
                                          channel_multiplier=channel_multiplier,
                                          padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        # output = tf.sigmoid(convolved)

        layers.append(convolved)

    print('3.shape: {}'.format(layers[-1].shape.as_list()))

    return layers[-1]


def resnet_d_1(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
               conv_type, channel_multiplier, padding):
    """
    Args:
      discrim_inputs: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      [N, 1]
    """
    layers = []
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        output = OptimizedResBlockDisc1(inputs, DIM_D=ndf, activation_fn='lrelu',
                                        spectral_normed=spectral_normed,
                                        update_collection=None,
                                        inputs_norm=False,
                                        biases=True)

        layers.append(output)
        print('D.layer_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    layer_specs = [
        ndf * 1,  # encoder_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf]
        ndf * 2,  # encoder_3: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        ndf * 4,  # encoder_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        ndf * 8,  # encoder_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
        ndf * 8,  # encoder_5: [batch, 16, 16, ndf * 8] => [batch, 8, 8, ndf * 8]
        ndf * 16,  # encoder_6: [batch, 8, 8, ndf * 8] => [batch, 4, 4, ndf * 16]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                                   name='D.Block.%d' % (len(layers) + 1),
                                   spectral_normed=spectral_normed,
                                   update_collection=update_collection,
                                   inputs_norm=False,
                                   resample='down', labels=None, biases=True, activation_fn='lrelu')

            if out_channels == ndf * 2:
                output, attn_score = Self_Atten(output, spectral_normed=spectral_normed)  # attention module
                print('Self_Atten.D: {}'.format(output.shape.as_list()))

            layers.append(output)
            print('D.layer_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    # layer_7: [batch, 4, 4, ngf * 16] => [batch, 4, 4, ngf * 16]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], ndf * 16, 3,
                               name='D.Block.%d' % (len(layers) + 1),
                               spectral_normed=spectral_normed,
                               update_collection=update_collection,
                               inputs_norm=False,
                               resample=None, labels=None, biases=True, activation_fn='lrelu')

        layers.append(output)
        print('D.layer_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    # layer_8: [batch, 4, 4, ndf * 16] => [batch, 4, 4, ndf * 16]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = norm_layer(layers[-1], decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
        output = nonlinearity(output, 'relu')

        # padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        # convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf * 8, 4, 1,
        #                                   name='Conv2D',
        #                                   conv_type=conv_type,
        #                                   channel_multiplier=channel_multiplier,
        #                                   padding=padding,
        #                                   spectral_normed=spectral_normed,
        #                                   update_collection=update_collection,
        #                                   inputs_norm=False,
        #                                   he_init=True, biases=True)

        output = tf.reduce_mean(output, axis=[1, 2])
        output = lib.ops.linear.Linear(
            output, ndf * 16, 1, 'D.Output',
            spectral_normed=True, update_collection=update_collection, biases=True, initialization='he')
        output = tf.reshape(output, [-1])

        layers.append(output)
        print('D.layer_{}: {}'.format(len(layers), layers[-1].shape.as_list()))

    return layers[-1]


# ######################## U-Net attention ######################## #

def unetResnetG(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                upsampe_method='depth_to_space'):
    """ UNet using ResNet architecture.
    Args:

    Returns:
    """
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = ResidualBlock(generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 3,
                               name='G.Block.%d' % (len(layers) + 1),
                               spectral_normed=True,
                               update_collection=None,
                               inputs_norm=False,
                               resample='down', labels=None, biases=True, activation_fn='relu')
        layers.append(output)
        print('G.shape: {}'.format(layers[-1].shape.as_list()))

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_9: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            output = ResidualBlock(layers[-1], layers[-1].shape.as_list()[-1], out_channels, 3,
                                   name='G.Block.%d' % (len(layers) + 1),
                                   spectral_normed=True,
                                   update_collection=None,
                                   inputs_norm=False,
                                   resample='down', labels=None, biases=True, activation_fn='relu')

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)
            print('G.shape: {}'.format(layers[-1].shape.as_list()))

    layer_specs = [
        (ngf * 8, 0.0),  # decoder_9: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_8: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_7: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_6: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 32, 32, ngf * 8 * 2] => [batch, 64, 64, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 64, 64, ngf * 4 * 2] => [batch, 128, 128, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 128, 128, ngf * 2 * 2] => [batch, 256, 256, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            output = ResidualBlock(inputs, inputs.shape.as_list()[-1], out_channels, 3,
                                   name='G.Block.%d' % (len(layers) + 1),
                                   spectral_normed=True,
                                   update_collection=None,
                                   inputs_norm=False,
                                   resample='up', labels=None, biases=True, activation_fn='relu')

            if decoder_layer in [5, 6]:
                output, attn_score = Self_Atten(output)  # attention module

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            print('G.shape: {}'.format(layers[-1].shape.as_list()))

    # decoder_1: [batch, 256, 256, ngf * 2] => [batch, 512, 512, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)

        output = ResidualBlock(inputs, inputs.shape.as_list()[-1], generator_outputs_channels, 3,
                               name='G.Block.%d' % (len(layers) + 1),
                               spectral_normed=True,
                               update_collection=None,
                               inputs_norm=False,
                               resample='up', labels=None, biases=True, activation_fn='relu')

        output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
        output = nonlinearity(output, 'relu')

        output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], generator_outputs_channels, 1,
                                       1, 'Conv2D',
                                       conv_type=conv_type,
                                       channel_multiplier=channel_multiplier,
                                       padding='SAME',
                                       spectral_normed=True, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        output = tf.nn.tanh(output)

        layers.append(output)

        print('G.output.shape: {}'.format(layers[-1].shape.as_list()))

    return layers[-1]


def unet_g(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
           upsampe_method='depth_to_space'):
    """ UNet.
    Args:

    Returns:
    """
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = lib.ops.conv2d.Conv2D(generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 4, 2, 'Conv2D',
                                       conv_type=conv_type,
                                       channel_multiplier=channel_multiplier,
                                       padding=padding,
                                       spectral_normed=True, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_9: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = nonlinearity(layers[-1], 'lrelu', 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = lib.ops.conv2d.Conv2D(rectified, rectified.shape.as_list()[-1], out_channels, 4, 2, 'Conv2D',
                                              conv_type=conv_type,
                                              channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=True, update_collection=None, inputs_norm=False,
                                              he_init=True, biases=True)

            output = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            # output = convolved

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.0),  # decoder_9: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_8: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_7: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_6: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 32, 32, ngf * 8 * 2] => [batch, 64, 64, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 64, 64, ngf * 4 * 2] => [batch, 128, 128, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 128, 128, ngf * 2 * 2] => [batch, 256, 256, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = nonlinearity(inputs, 'relu')
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            _b, h, w, _c = rectified.shape
            if upsampe_method == 'resize':
                resized_input = tf.image.resize_images(rectified, [h * 2, w * 2],
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif upsampe_method == 'depth_to_space':
                resized_input = tf.concat([rectified, rectified, rectified, rectified], axis=3)
                resized_input = tf.depth_to_space(resized_input, block_size=2)
            else:
                raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

            output = lib.ops.conv2d.Conv2D(resized_input, resized_input.shape.as_list()[-1], out_channels, 4, 1,
                                           'Conv2D',
                                           conv_type=conv_type,
                                           channel_multiplier=channel_multiplier,
                                           padding=padding,
                                           spectral_normed=True, update_collection=None, inputs_norm=False,
                                           he_init=True, biases=True)
            # output = tf.layers.conv2d_transpose(rectified, out_channels, kernel_size=4, strides=(2, 2),
            #                                     padding="same",
            #                                     kernel_initializer=tf.contrib.layers.xavier_initializer)

            output = norm_layer(output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")

            if decoder_layer in [6]:
                output, attn_score = Self_Atten(output)  # attention module
                print('Self_Atten.G: {}'.format(output.shape.as_list()))

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 256, 256, ngf * 2] => [batch, 512, 512, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = nonlinearity(inputs, 'relu')
        _b, h, w, _c = rectified.shape
        if upsampe_method == 'resize':
            resized_input = tf.image.resize_images(rectified, [h * 2, w * 2],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif upsampe_method == 'depth_to_space':
            resized_input = tf.concat([rectified, rectified, rectified, rectified], axis=3)
            resized_input = tf.depth_to_space(resized_input, block_size=2)
        else:
            raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

        output = lib.ops.conv2d.Conv2D(resized_input, resized_input.shape.as_list()[-1], generator_outputs_channels, 4,
                                       1, 'Conv2D',
                                       conv_type=conv_type,
                                       channel_multiplier=channel_multiplier,
                                       padding=padding,
                                       spectral_normed=True, update_collection=None, inputs_norm=False,
                                       he_init=True, biases=True)
        output = tf.nn.tanh(output)

        layers.append(output)

    return layers[-1]


def unet_d(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
           conv_type, channel_multiplier, padding):
    """
    Args:
      discrim_inputs: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      [N, 30, 30, ndf]
    """
    n_layers = 4
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf, 4, 2,
                                          'Conv2D',
                                          conv_type=conv_type,
                                          channel_multiplier=channel_multiplier,
                                          padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        layers.append(rectified)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], out_channels_, 4, stride,
                                              'Conv2D',
                                              conv_type=conv_type,
                                              channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=spectral_normed,
                                              update_collection=update_collection,
                                              inputs_norm=False,
                                              he_init=True, biases=True)

            # normalized = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            normalized = convolved
            rectified = nonlinearity(normalized, 'lrelu', 0.2)

            if i in [0]:
                rectified, attn_score = Self_Atten(rectified)  # attention module
                print('Self_Atten.D: {}'.format(rectified.shape.as_list()))

            layers.append(rectified)

    # layer_6: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(rectified, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], 1, 4, 1,
                                          'Conv2D',
                                          conv_type=conv_type,
                                          channel_multiplier=channel_multiplier,
                                          padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        # output = tf.sigmoid(convolved)
        output = convolved

        layers.append(output)

    return layers[-1]


# TODO
def unet_d_(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
            conv_type, channel_multiplier, padding):
    """
    Args:
      discrim_inputs: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      [N, 1]
    """

    n_layers = 5
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf, 4, 2,
                                          'Conv2D',
                                          conv_type=conv_type,
                                          channel_multiplier=channel_multiplier,
                                          padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        layers.append(rectified)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    # layer_6: [batch, 16, 16, ndf * 8] => [batch, 8, 8, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            # stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            stride = 2
            padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], out_channels_, 4, stride,
                                              'Conv2D',
                                              conv_type=conv_type,
                                              channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=spectral_normed,
                                              update_collection=update_collection,
                                              inputs_norm=False,
                                              he_init=True, biases=True)

            # normalized = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            rectified = nonlinearity(convolved, 'lrelu', 0.2)

            if i in [0, 1]:
                rectified, attn_score = Self_Atten(rectified)  # attention module
                print('Self_Atten.D: {}'.format(rectified.shape.as_list()))

            layers.append(rectified)

    # layer_7: [batch, 8, 8, ndf * 8] => [batch, 7, 7, ndf * 2]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(rectified, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf * 2, 4, 1,
                                       'Conv2D',
                                       conv_type=conv_type,
                                       channel_multiplier=channel_multiplier,
                                       padding=padding,
                                       spectral_normed=spectral_normed,
                                       update_collection=update_collection,
                                       inputs_norm=False,
                                       he_init=True, biases=True)
        # output = tf.sigmoid(convolved)

        layers.append(output)

    # layer_8: [batch, 7, 7, ndf * 2] => [batch, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = nonlinearity(output, 'lrelu', 0.2)
        output = tf.reduce_mean(output, axis=[1, 2])
        output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output',
                                       spectral_normed=spectral_normed,
                                       update_collection=update_collection)
        output = tf.reshape(output, [-1])

        layers.append(output)

    return layers[-1]


# ######################## U-Net ######################## #

def unet_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                   upsampe_method='depth_to_space'):
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = lib.ops.conv2d.Conv2D(
            generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 4, 2, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier,
            padding=padding, spectral_normed=True, update_collection=None,
            inputs_norm=False, he_init=True, biases=True)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # ngf * 8,  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = nonlinearity(layers[-1], 'lrelu', 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = lib.ops.conv2d.Conv2D(
                rectified, rectified.shape.as_list()[-1], out_channels, 4, 2, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier,
                padding=padding, spectral_normed=True, update_collection=None,
                inputs_norm=False, he_init=True, biases=True)

            output = norm_layer(convolved, decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)

    layer_specs = [
        # (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        # (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 32, 32, ngf * 8 * 2] => [batch, 64, 64, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 64, 64, ngf * 4 * 2] => [batch, 128, 128, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 128, 128, ngf * 2 * 2] => [batch, 256, 256, ngf * 2]
    ]
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = nonlinearity(inputs, 'relu')
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            _, h, w, _ = rectified.shape
            if upsampe_method == 'resize':
                resized_input = tf.image.resize_images(
                    rectified, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif upsampe_method == 'depth_to_space':
                resized_input = tf.concat([rectified, rectified, rectified, rectified], axis=3)
                resized_input = tf.depth_to_space(resized_input, block_size=2)
            else:
                raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

            output = lib.ops.conv2d.Conv2D(
                resized_input, resized_input.shape.as_list()[-1], out_channels, 4, 1, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier,
                padding=padding, spectral_normed=True, update_collection=None,
                inputs_norm=False, he_init=True, biases=True)
            # output = tf.layers.conv2d_transpose(rectified, out_channels, kernel_size=4, strides=(2, 2),
            #                                     padding="same",
            #                                     kernel_initializer=tf.contrib.layers.xavier_initializer)

            output = norm_layer(output, decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")

            # if dropout > 0.0:
            #     output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 256, 256, ngf * 2] => [batch, 512, 512, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = nonlinearity(inputs, 'relu')
        _, h, w, _ = rectified.shape
        if upsampe_method == 'resize':
            resized_input = tf.image.resize_images(
                rectified, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif upsampe_method == 'depth_to_space':
            resized_input = tf.concat([rectified, rectified, rectified, rectified], axis=3)
            resized_input = tf.depth_to_space(resized_input, block_size=2)
        else:
            raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

        output = lib.ops.conv2d.Conv2D(
            resized_input, resized_input.shape.as_list()[-1], generator_outputs_channels, 4, 1, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier,
            padding=padding, spectral_normed=True, update_collection=None,
            inputs_norm=False, he_init=True, biases=True)
        output = tf.nn.tanh(output)

        layers.append(output)

    return layers[-1]


def unet_generator_1(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                     upsampe_method='depth_to_space'):
    """ Try different channels and kernel size.
    Args:

    Returns:
    """
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = lib.ops.conv2d.Conv2D(
            generator_inputs, generator_inputs.shape.as_list()[-1], ngf, 5, 2, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
            spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 16,  # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 16,  # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # ngf * 8,  # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            output = lib.ops.conv2d.Conv2D(
                layers[-1], layers[-1].shape.as_list()[-1], out_channels, 5, 2, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)

            output = norm_layer(output, decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")
            output = nonlinearity(output, 'lrelu', 0.2)

            # output, attn_score = Self_Attn(output)  # attention module

            layers.append(output)

    layer_specs = [
        # (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        # (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 16, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 32, 32, ngf * 8 * 2] => [batch, 64, 64, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 64, 64, ngf * 4 * 2] => [batch, 128, 128, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 128, 128, ngf * 2 * 2] => [batch, 256, 256, ngf * 2]
    ]
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                inputs = layers[-1]
            else:
                inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            _, h, w, _ = inputs.shape
            if upsampe_method == 'resize':
                inputs = tf.image.resize_images(
                    inputs, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif upsampe_method == 'depth_to_space':
                inputs = tf.concat([inputs, inputs, inputs, inputs], axis=3)
                inputs = tf.depth_to_space(inputs, block_size=2)
            # else:
            #     raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

            output = lib.ops.conv2d.Conv2D(
                inputs, inputs.shape.as_list()[-1], out_channels, 5, 1, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)
            # resized_input = tf.layers.conv2d_transpose(
            #     inputs, out_channels, kernel_size=4, strides=(2, 2), padding="same",
            #     kernel_initializer=tf.initializers.he_normal())

            output = norm_layer(output, decay=0.9, epsilon=1e-6, is_training=True, norm_type="IN")
            output = nonlinearity(output, 'relu')

            # if dropout > 0.0:
            #     output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 256, 256, ngf * 2] => [batch, 512, 512, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        inputs = tf.concat([layers[-1], layers[0]], axis=3)
        # rectified = nonlinearity(inputs, 'relu')
        _, h, w, _ = inputs.shape
        if upsampe_method == 'resize':
            inputs = tf.image.resize_images(
                inputs, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif upsampe_method == 'depth_to_space':
            inputs = tf.concat([inputs, inputs, inputs, inputs], axis=3)
            inputs = tf.depth_to_space(inputs, block_size=2)
        # else:
        #     raise NotImplementedError('upsampe_method [%s] is not recognized' % upsampe_method)

        output = lib.ops.conv2d.Conv2D(
            inputs, inputs.shape.as_list()[-1], generator_outputs_channels, 5, 1, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
            spectral_normed=True, update_collection=None, inputs_norm=False, he_init=True, biases=True)
        output = tf.nn.tanh(output)

        layers.append(output)

    return layers[-1]


def unet_discriminator(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                       conv_type, channel_multiplier, padding):
    n_layers = 4
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        convolved = lib.ops.conv2d.Conv2D(
            padded_input, padded_input.shape.as_list()[-1], ndf, 4, 2, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier,
            padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
            inputs_norm=False, he_init=True, biases=True)
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        layers.append(rectified)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            convolved = lib.ops.conv2d.Conv2D(
                padded_input, padded_input.shape.as_list()[-1], out_channels_, 4, stride, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier,
                padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
                inputs_norm=False, he_init=True, biases=True)

            # convolved = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            rectified = nonlinearity(convolved, 'lrelu', 0.2)

            layers.append(rectified)

    # layer_6: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(rectified, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        convolved = lib.ops.conv2d.Conv2D(
            padded_input, padded_input.shape.as_list()[-1], 1, 4, 1, 'Conv2D', conv_type, channel_multiplier,
            padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
            inputs_norm=False, he_init=True, biases=True)
        # convolved = tf.sigmoid(convolved)
        layers.append(convolved)

    return layers[-1]


def unet_discriminator_1(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                         conv_type, channel_multiplier, padding):
    """Corresponding to unet_generator_1.
    Compared to unet_discriminator: more out channels, big kernel size.
    Args:

    Returns:
    """
    n_layers = 4
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        padded_input = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        convolved = lib.ops.conv2d.Conv2D(
            padded_input, padded_input.shape.as_list()[-1], ndf, 5, 2, 'Conv2D',
            conv_type=conv_type, channel_multiplier=channel_multiplier,
            padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
            inputs_norm=False, he_init=True, biases=True)
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        layers.append(rectified)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 8] => [batch, 31, 31, ndf * 16]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * (2 ** (i + 1))
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            padded_input = tf.pad(layers[-1], [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
            convolved = lib.ops.conv2d.Conv2D(
                padded_input, padded_input.shape.as_list()[-1], out_channels_, 5, stride, 'Conv2D',
                conv_type=conv_type, channel_multiplier=channel_multiplier,
                padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
                inputs_norm=False, he_init=True, biases=True)

            # convolved = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            rectified = nonlinearity(convolved, 'lrelu', 0.2)

            layers.append(rectified)

    # layer_6: [batch, 31, 31, ndf * 16] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(rectified, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
        convolved = lib.ops.conv2d.Conv2D(
            padded_input, padded_input.shape.as_list()[-1], 1, 5, 1, 'Conv2D', conv_type, channel_multiplier,
            padding=padding, spectral_normed=spectral_normed, update_collection=update_collection,
            inputs_norm=False, he_init=True, biases=True)
        # convolved = tf.sigmoid(convolved)
        layers.append(convolved)

    return layers[-1]


def unet_discriminator_(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                        conv_type, channel_multiplier, padding):
    """Output is: [batch, 14, 14, 1]
    Args:

    Returns:
    """
    n_layers = 5
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 512, 512, in_channels * 2] => [batch, 256, 256, ndf]
    with tf.variable_scope("layer_1"):
        padded_input = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], ndf, 4, 2,
                                          'Conv2D',
                                          conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        rectified = nonlinearity(convolved, 'lrelu', 0.2)

        layers.append(rectified)

    # layer_2: [batch, 256, 256, ndf] => [batch, 128, 128, ndf * 2]
    # layer_3: [batch, 128, 128, ndf * 2] => [batch, 64, 64, ndf * 4]
    # layer_4: [batch, 64, 64, ndf * 4] => [batch, 32, 32, ndf * 8]
    # layer_5: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    # layer_6: [batch, 16, 16, ndf * 4] => [batch, 15, 15, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels_ = ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], out_channels_, 4, stride,
                                              'Conv2D',
                                              conv_type=conv_type, channel_multiplier=channel_multiplier,
                                              padding=padding,
                                              spectral_normed=spectral_normed,
                                              update_collection=update_collection,
                                              inputs_norm=False,
                                              he_init=True, biases=True)

            # normalized = norm_layer(convolved, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
            rectified = nonlinearity(convolved, 'lrelu', 0.2)

            layers.append(rectified)

    # layer_7: [batch, 15, 15, ndf * 8] => [batch, 14, 14, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        padded_input = tf.pad(rectified, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = lib.ops.conv2d.Conv2D(padded_input, padded_input.shape.as_list()[-1], 1, 4, 1,
                                          'Conv2D',
                                          conv_type=conv_type, channel_multiplier=channel_multiplier, padding=padding,
                                          spectral_normed=spectral_normed,
                                          update_collection=update_collection,
                                          inputs_norm=False,
                                          he_init=True, biases=True)
        # output = tf.sigmoid(convolved)
        output = convolved

        layers.append(output)

    return layers[-1]


# ######################## VGG ######################## #

VGG_MEAN = [103.939, 116.779, 123.68]


def vgg_generator(generator_inputs, generator_outputs_channels, ngf, conv_type, channel_multiplier, padding,
                  train_mode=None, trainable=None, vgg19_npy_path=None):
    """vgg generator

    Args:
      generator_inputs: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      generator_outputs_channels:
      ngf:
      conv_type:
      channel_multiplier:
      padding:
      train_mode:
      trainable:

    Returns:
      Returns generated image batch.
    """
    if vgg19_npy_path is not None:
        data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    else:
        data_dict = None

    rgb_scaled = (generator_inputs + 1) / 2  # [-1, 1] => [0, 1]
    rgb_scaled *= 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(value=rgb_scaled, num_or_size_splits=3, axis=3)
    # assert red.get_shape().as_list()[1:] == [224, 224, 1]
    # assert green.get_shape().as_list()[1:] == [224, 224, 1]
    # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ], axis=3)
    # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = conv_layer(bgr, 6, 64, "conv1_1", trainable=True, data_dict=None)
    conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2", trainable=False, data_dict=data_dict)
    pool1 = max_pool(conv1_2, 'pool1')  # [112, 112, 64], [256, 256, 64]

    conv2_1 = conv_layer(pool1, 64, 128, "conv2_1", trainable=False, data_dict=data_dict)
    conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2", trainable=False, data_dict=data_dict)
    pool2 = max_pool(conv2_2, 'pool2')  # [56, 56, 128], [128, 128, 128]

    conv3_1 = conv_layer(pool2, 128, 256, "conv3_1", trainable=False, data_dict=data_dict)
    conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2", trainable=False, data_dict=data_dict)
    conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3", trainable=False, data_dict=data_dict)
    conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4", trainable=False, data_dict=data_dict)
    pool3 = max_pool(conv3_4, 'pool3')  # [28, 28, 256], [64, 64, 256]

    conv4_1 = conv_layer(pool3, 256, 512, "conv4_1", trainable=True, data_dict=data_dict)
    conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2", trainable=True, data_dict=data_dict)
    conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3", trainable=True, data_dict=data_dict)
    conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4", trainable=True, data_dict=data_dict)
    pool4 = max_pool(conv4_4, 'pool4')  # [14, 14, 512], [32, 32, 512]

    conv5_1 = conv_layer(pool4, 512, 512, "conv5_1", trainable=True, data_dict=data_dict)
    conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2", trainable=True, data_dict=data_dict)
    conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3", trainable=True, data_dict=data_dict)
    conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4", trainable=True, data_dict=data_dict)
    pool5 = max_pool(conv5_4, 'pool5')  # [7, 7, 512], [16, 16, 512]

    # fc6 = fc_layer(pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    # relu6 = tf.nn.relu(fc6)
    # if train_mode is not None:
    #     relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(relu6, 0.5), lambda: relu6)
    # elif trainable:
    #     relu6 = tf.nn.dropout(relu6, 0.5)
    #
    # fc7 = fc_layer(relu6, 4096, 4096, "fc7")
    # relu7 = tf.nn.relu(fc7)
    # if train_mode is not None:
    #     relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(relu7, 0.5), lambda: relu7)
    # elif trainable:
    #     relu7 = tf.nn.dropout(relu7, 0.5)
    #
    # fc8 = fc_layer(relu7, 4096, 1000, "fc8")
    #
    # prob = tf.nn.softmax(fc8, name="prob")
    #
    # data_dict = None

    # ################ add other layers ################ #
    conv6_1 = lib.ops.conv2d.Conv2D(pool5, pool5.shape.as_list()[-1], 512, 3, 2, 'conv6_1',
                                    conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                    spectral_normed=False, update_collection=None, inputs_norm=False,
                                    he_init=True, biases=True)  # [8, 8, 512]
    conv6_1 = norm_layer(conv6_1, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv6_1 = nonlinearity(conv6_1, 'relu', 0.2)

    conv6_2 = lib.ops.conv2d.Conv2D(conv6_1, conv6_1.shape.as_list()[-1], 512, 3, 2, 'conv6_2',
                                    conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                    spectral_normed=False, update_collection=None, inputs_norm=False,
                                    he_init=True, biases=True)  # [4, 4, 512]
    conv6_2 = norm_layer(conv6_2, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv6_2 = nonlinearity(conv6_2, 'relu', 0.2)

    # decoder part
    conv6_2_decoder = tf.concat([conv6_2, conv6_2, conv6_2, conv6_2], axis=3, name="conv6_2_concat")
    conv6_2_decoder = tf.depth_to_space(conv6_2_decoder, 2, name='conv6_2_decoder')  # [8, 8, 512]
    conv6_2_decoder = lib.ops.conv2d.Conv2D(conv6_2_decoder, conv6_2_decoder.shape.as_list()[-1], 512, 1, 1,
                                            "conv6_2_conv",
                                            spectral_normed=False,
                                            update_collection=None,
                                            he_init=True, biases=True)
    conv6_2_decoder = norm_layer(conv6_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv6_2_decoder = nonlinearity(conv6_2_decoder, 'relu', 0.2)

    conv6_2_decoder = tf.concat([conv6_2_decoder, conv6_1], axis=3)
    conv6_1_decoder = tf.concat([conv6_2_decoder, conv6_2_decoder, conv6_2_decoder, conv6_2_decoder], axis=3,
                                name="conv6_1_concat")
    conv6_1_decoder = tf.depth_to_space(conv6_1_decoder, 2, name='conv6_1_decoder')  # [16, 16, 512*2]
    conv6_1_decoder = lib.ops.conv2d.Conv2D(conv6_1_decoder, conv6_1_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv6_1_conv",
                                            spectral_normed=False,
                                            update_collection=None,
                                            he_init=True, biases=True)
    conv6_1_decoder = norm_layer(conv6_1_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv6_1_decoder = nonlinearity(conv6_1_decoder, 'relu', 0.2)

    conv6_1_decoder = tf.concat([conv6_1_decoder, pool5], axis=3)
    pool5_decoder = tf.concat([conv6_1_decoder, conv6_1_decoder, conv6_1_decoder, conv6_1_decoder], axis=3,
                              name="pool5_decoder_concat")
    pool5_decoder = tf.depth_to_space(pool5_decoder, 2, name='pool5_decoder')  # [32, 32, 512*2]
    conv5_4_decoder = lib.ops.conv2d.Conv2D(pool5_decoder, pool5_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_4_decoder", he_init=True, biases=True)
    conv5_4_decoder = norm_layer(conv5_4_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv5_4_decoder = nonlinearity(conv5_4_decoder, 'relu', 0.2)
    conv5_3_decoder = lib.ops.conv2d.Conv2D(conv5_4_decoder, conv5_4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_3_decoder", he_init=True, biases=True)
    conv5_3_decoder = norm_layer(conv5_3_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv5_3_decoder = nonlinearity(conv5_3_decoder, 'relu', 0.2)
    conv5_2_decoder = lib.ops.conv2d.Conv2D(conv5_3_decoder, conv5_3_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_2_decoder", he_init=True, biases=True)
    conv5_2_decoder = norm_layer(conv5_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv5_2_decoder = nonlinearity(conv5_2_decoder, 'relu', 0.2)
    conv5_1_decoder = lib.ops.conv2d.Conv2D(conv5_2_decoder, conv5_2_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv5_1_decoder", he_init=True, biases=True)
    conv5_1_decoder = norm_layer(conv5_1_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv5_1_decoder = nonlinearity(conv5_1_decoder, 'relu', 0.2)

    conv5_1_decoder = tf.concat([conv5_1_decoder, pool4], axis=3)
    pool4_decoder = tf.concat([conv5_1_decoder, conv5_1_decoder, conv5_1_decoder, conv5_1_decoder], axis=3,
                              name="pool4_decoder_concat")
    pool4_decoder = tf.depth_to_space(pool4_decoder, 2, name='pool4_decoder')  # [64, 64, 512*2]
    conv4_4_decoder = lib.ops.conv2d.Conv2D(pool4_decoder, pool4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_4_decoder", he_init=True, biases=True)
    conv4_4_decoder = norm_layer(conv4_4_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv4_4_decoder = nonlinearity(conv4_4_decoder, 'relu', 0.2)
    conv4_3_decoder = lib.ops.conv2d.Conv2D(conv4_4_decoder, conv4_4_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_3_decoder", he_init=True, biases=True)
    conv4_3_decoder = norm_layer(conv4_3_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv4_3_decoder = nonlinearity(conv4_3_decoder, 'relu', 0.2)
    conv4_2_decoder = lib.ops.conv2d.Conv2D(conv4_3_decoder, conv4_3_decoder.shape.as_list()[-1], 512, 3, 1,
                                            "conv4_2_decoder", he_init=True, biases=True)
    conv4_2_decoder = norm_layer(conv4_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv4_2_decoder = nonlinearity(conv4_2_decoder, 'relu', 0.2)
    conv4_1_decoder = lib.ops.conv2d.Conv2D(conv4_2_decoder, conv4_2_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv4_1_decoder", he_init=True, biases=True)
    conv4_1_decoder = norm_layer(conv4_1_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv4_1_decoder = nonlinearity(conv4_1_decoder, 'relu', 0.2)

    conv4_1_decoder = tf.concat([conv4_1_decoder, pool3], axis=3)
    pool3_decoder = tf.concat([conv4_1_decoder, conv4_1_decoder, conv4_1_decoder, conv4_1_decoder], axis=3,
                              name="pool3_decoder_concat")
    pool3_decoder = tf.depth_to_space(pool3_decoder, 2, name='pool3_decoder')  # [128, 128, 256*2]
    conv3_4_decoder = lib.ops.conv2d.Conv2D(pool3_decoder, pool3_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_4_decoder", he_init=True, biases=True)
    conv3_4_decoder = norm_layer(conv3_4_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv3_4_decoder = nonlinearity(conv3_4_decoder, 'relu', 0.2)
    conv3_3_decoder = lib.ops.conv2d.Conv2D(conv3_4_decoder, conv3_4_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_3_decoder", he_init=True, biases=True)
    conv3_3_decoder = norm_layer(conv3_3_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv3_3_decoder = nonlinearity(conv3_3_decoder, 'relu', 0.2)
    conv3_2_decoder = lib.ops.conv2d.Conv2D(conv3_3_decoder, conv3_3_decoder.shape.as_list()[-1], 256, 3, 1,
                                            "conv3_2_decoder", he_init=True, biases=True)
    conv3_2_decoder = norm_layer(conv3_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv3_2_decoder = nonlinearity(conv3_2_decoder, 'relu', 0.2)
    conv3_1_decoder = lib.ops.conv2d.Conv2D(conv3_2_decoder, conv3_2_decoder.shape.as_list()[-1], 128, 3, 1,
                                            "conv3_1_decoder", he_init=True, biases=True)
    conv3_1_decoder = norm_layer(conv3_1_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv3_1_decoder = nonlinearity(conv3_1_decoder, 'relu', 0.2)

    conv3_1_decoder = tf.concat([conv3_1_decoder, pool2], axis=3)
    pool2_decoder = tf.concat([conv3_1_decoder, conv3_1_decoder, conv3_1_decoder, conv3_1_decoder], axis=3,
                              name="pool2_decoder_concat")
    pool2_decoder = tf.depth_to_space(pool2_decoder, 2, name='pool2_decoder')  # [256, 256, 128*2]
    conv2_2_decoder = lib.ops.conv2d.Conv2D(pool2_decoder, pool2_decoder.shape.as_list()[-1], 128, 3, 1,
                                            "conv2_2_decoder", he_init=True, biases=True)
    conv2_2_decoder = norm_layer(conv2_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv2_2_decoder = nonlinearity(conv2_2_decoder, 'relu', 0.2)
    conv2_1_decoder = lib.ops.conv2d.Conv2D(conv2_2_decoder, conv2_2_decoder.shape.as_list()[-1], 64, 3, 1,
                                            "conv2_1_decoder", he_init=True, biases=True)
    conv2_1_decoder = norm_layer(conv2_1_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv2_1_decoder = nonlinearity(conv2_1_decoder, 'relu', 0.2)

    conv2_1_decoder = tf.concat([conv2_1_decoder, pool1], axis=3)
    pool1_decoder = tf.concat([conv2_1_decoder, conv2_1_decoder, conv2_1_decoder, conv2_1_decoder], axis=3,
                              name="pool1_decoder_concat")
    pool1_decoder = tf.depth_to_space(pool1_decoder, 2, name='pool1_decoder')  # [512, 512, 64*2]
    conv1_2_decoder = lib.ops.conv2d.Conv2D(pool1_decoder, pool1_decoder.shape.as_list()[-1], 64, 3, 1,
                                            "conv1_2_decoder", he_init=True, biases=True)
    conv1_2_decoder = norm_layer(conv1_2_decoder, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")
    conv1_2_decoder = nonlinearity(conv1_2_decoder, 'relu', 0.2)

    conv1_2_decoder = tf.concat([conv1_2_decoder, bgr], axis=3)
    bgr_output = lib.ops.conv2d.Conv2D(conv1_2_decoder, conv1_2_decoder.shape.as_list()[-1], 3, 3, 1,
                                       "bgr_output", he_init=True, biases=True)
    bgr_output = norm_layer(bgr_output, decay=0.9, epsilon=1e-5, is_training=True, norm_type="IN")

    bgr_output = tf.nn.sigmoid(bgr_output)

    # # convert bgr to rgb
    # b, g, r = tf.split(bgr_output, 3, axis=3)
    # rgb = tf.concat([r, g, b], axis=3)

    return bgr_output


def vgg_discriminator(discrim_inputs, discrim_targets, ndf, spectral_normed, update_collection,
                      conv_type, channel_multiplier, padding):
    """vgg generator

    Args:
      discrim_inputs: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      discrim_targets:
      ndf:
      spectral_normed:
      update_collection:
      conv_type:
      channel_multiplier:
      padding:

    Returns:
      Returns generated image batch.
    """

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    inputs = tf.concat([discrim_inputs, discrim_targets], axis=3)
    print('inputs.shape: {}'.format(inputs.shape.as_list()))

    # block 1
    output = lib.ops.conv2d.Conv2D(inputs, inputs.shape.as_list()[-1], 32, 1, 1, 'D_conv1',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [512, 512, 32]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv2',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [512, 512, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [256, 256, 64]
    print('block 1.shape: {}'.format(inputs.shape.as_list()))

    # block 2
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv3',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [256, 256, 64]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv4',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [256, 256, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [128, 128, 64]
    print('block 2.shape: {}'.format(inputs.shape.as_list()))

    # block 3
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv5',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [128, 128, 64]
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D(output, output.shape.as_list()[-1], 64, 3, 1, 'D_conv6',
                                   conv_type=conv_type, channel_multiplier=channel_multiplier, padding='SAME',
                                   spectral_normed=True, update_collection=update_collection, inputs_norm=False,
                                   he_init=True, biases=True)  # [128, 128, 64]
    output = tf.nn.relu(output)
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # [64, 64, 64]
    print('block 3.shape: {}'.format(inputs.shape.as_list()))

    # fc layer
    output = tf.reshape(output, [output.shape.as_list()[0], -1])
    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 100, 'D.fc1',
                                   spectral_normed=True, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    output = tf.nn.tanh(output)
    print('fc1.shape: {}'.format(inputs.shape.as_list()))

    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 2, 'D.fc2',
                                   spectral_normed=True, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    output = tf.nn.tanh(output)
    print('fc2.shape: {}'.format(inputs.shape.as_list()))

    output = lib.ops.linear.Linear(output, output.shape.as_list()[-1], 1, 'D.Output',
                                   spectral_normed=True, update_collection=update_collection,
                                   inputs_norm=False,
                                   biases=True, initialization=None, weightnorm=None, gain=1.)
    # output = tf.nn.sigmoid(output)
    print('Output.shape: {}'.format(inputs.shape.as_list()))

    return output


# ######################## helper function for VGG ######################## #
def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, in_channels, out_channels, name, trainable=True, data_dict=None):
    # with tf.variable_scope(name):
    filt, conv_biases = get_conv_var(3, in_channels, out_channels, name,
                                     trainable=trainable, data_dict=data_dict)

    conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, conv_biases)
    relu = tf.nn.relu(bias)

    return relu


def fc_layer(bottom, in_size, out_size, name, trainable=True, data_dict=None):
    # with tf.variable_scope(name):
    weights, biases = get_fc_var(in_size, out_size, name, trainable=trainable, data_dict=data_dict)

    x = tf.reshape(bottom, [-1, in_size])
    fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    return fc


def get_conv_var(filter_size, in_channels, out_channels, name, trainable=True, data_dict=None):
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name, 0, name + "_filters", trainable=trainable, data_dict=data_dict)

    initial_value = tf.truncated_normal([out_channels], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases", trainable=trainable, data_dict=data_dict)

    return filters, biases


def get_fc_var(in_size, out_size, name, trainable=True, data_dict=None):
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    weights = get_var(initial_value, name, 0, name + "_weights", trainable=trainable, data_dict=data_dict)

    initial_value = tf.truncated_normal([out_size], .0, .001)
    biases = get_var(initial_value, name, 1, name + "_biases", trainable=trainable, data_dict=data_dict)

    return weights, biases


def get_var(initial_value, name, idx, var_name, trainable=True, data_dict=None):
    """vgg generator

    Args:
      initial_value: A batch of images to translate.
        Images should be normalized already. Shape is [batch, height, width, channels].
      name:
      idx:
      var_name:
      trainable:
      data_dict: Pre-trained variables.

    Returns:
      Returns generated image batch.
    """
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    # var_dict[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var

# def save_npy(sess, npy_path="./vgg19-save.npy"):
#     assert isinstance(sess, tf.Session)
#
#     data_dict = {}
#
#     for (name, idx), var in list(var_dict.items()):
#         var_out = sess.run(var)
#         if name not in data_dict:
#             data_dict[name] = {}
#         data_dict[name][idx] = var_out
#
#     np.save(npy_path, data_dict)
#     print(("file saved", npy_path))
#     return npy_path
#
#
# def get_var_count():
#     count = 0
#     for v in list(var_dict.values()):
#         count += reduce(lambda x, y: x * y, v.get_shape().as_list())
#     return count
