# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Networks for GAN Pix2Pix example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

layers = tf.contrib.layers
import collections


FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'


def cyclegan_arg_scope(instance_norm_center=True,
                       instance_norm_scale=True,
                       instance_norm_epsilon=0.001,
                       weights_init_stddev=0.02,
                       weight_decay=0.0):
    """Returns a default argument scope for all generators and discriminators.

    Args:
      instance_norm_center: Whether instance normalization applies centering.
      instance_norm_scale: Whether instance normalization applies scaling.
      instance_norm_epsilon: Small float added to the variance in the instance
        normalization to avoid dividing by zero.
      weights_init_stddev: Standard deviation of the random values to initialize
        the convolution kernels with.
      weight_decay: Magnitude of weight decay applied to all convolution kernel
        variables of the generator.

    Returns:
      An arg-scope.
    """
    instance_norm_params = {
        'center': instance_norm_center,
        'scale': instance_norm_scale,
        'epsilon': instance_norm_epsilon,
    }

    weights_regularizer = None
    if weight_decay and weight_decay > 0.0:
        weights_regularizer = layers.l2_regularizer(weight_decay)

    with tf.contrib.framework.arg_scope(
            [layers.conv2d],
            normalizer_fn=layers.instance_norm,
            normalizer_params=instance_norm_params,
            weights_initializer=tf.random_normal_initializer(0, weights_init_stddev),
            weights_regularizer=weights_regularizer) as sc:
        return sc


def pix2pix_arg_scope():
    """Returns a default argument scope for isola_net.

    Returns:
      An arg scope.
    """
    # These parameters come from the online port, which don't necessarily match
    # those in the paper.
    # TODO(nsilberman): confirm these values with Philip.
    instance_norm_params = {
        'center': True,
        'scale': True,
        'epsilon': 0.00001,
    }

    with tf.contrib.framework.arg_scope(
            [layers.conv2d, layers.conv2d_transpose],
            normalizer_fn=layers.instance_norm,
            normalizer_params=instance_norm_params,
            weights_initializer=tf.random_normal_initializer(0, 0.02)) as sc:
        return sc


def upsample(net, num_outputs, kernel_size, method='nn_upsample_conv'):
    """Upsamples the given inputs.

    Args:
      net: A `Tensor` of size [batch_size, height, width, filters].
      num_outputs: The number of output filters.
      kernel_size: A list of 2 scalars or a 1x2 `Tensor` indicating the scale,
        relative to the inputs, of the output dimensions. For example, if kernel
        size is [2, 3], then the output height and width will be twice and three
        times the input size.
      method: The upsampling method.

    Returns:
      An `Tensor` which was upsampled using the specified method.

    Raises:
      ValueError: if `method` is not recognized.
    """
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]

    if method == 'nn_upsample_conv':
        net = tf.image.resize_nearest_neighbor(
            net, [kernel_size[0] * height, kernel_size[1] * width])
        net = layers.conv2d(net, num_outputs, [4, 4], activation_fn=None)
    elif method == 'conv2d_transpose':
        net = layers.conv2d_transpose(
            net, num_outputs, [4, 4], stride=kernel_size, activation_fn=None)
    else:
        raise ValueError('Unknown method: [%s]' % method)

    return net


class Block(
    collections.namedtuple('Block', ['num_filters', 'decoder_keep_prob'])):
    """Represents a single block of encoder and decoder processing.

    The Image-to-Image translation paper works a bit differently than the original
    U-Net model. In particular, each block represents a single operation in the
    encoder which is concatenated with the corresponding decoder representation.
    A dropout layer follows the concatenation and convolution of the concatenated
    features.
    """
    pass


def _default_generator_blocks():
    """Returns the default generator block definitions.

    Returns:
      A list of generator blocks.
    """
    return [
        Block(64, 0.5),
        Block(128, 0.5),
        Block(256, 0.5),
        Block(512, 0),
        Block(512, 0),
        Block(512, 0),
        Block(512, 0),
    ]


def pix2pix_discriminator(net, num_filters, padding=2, pad_mode='REFLECT',
                          activation_fn=tf.nn.leaky_relu, is_training=False):
    """Creates the Image2Image Translation Discriminator.

    Args:
      net: A `Tensor` of size [batch_size, height, width, channels] representing
        the input.
      num_filters: A list of the filters in the discriminator. The length of the
        list determines the number of layers in the discriminator.
      padding: Amount of reflection padding applied before each convolution.
      pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".
      activation_fn: activation fn for layers.conv2d.
      is_training: Whether or not the model is training or testing.

    Returns:
      A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
      'patches' we're attempting to discriminate and a dictionary of model end
      points.
    """
    del is_training
    end_points = {}

    num_layers = len(num_filters)

    def padded(net, scope):
        if padding:
            with tf.variable_scope(scope):
                spatial_pad = tf.constant(
                    [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                    dtype=tf.int32)
                return tf.pad(net, spatial_pad, pad_mode)
        else:
            return net

    with tf.contrib.framework.arg_scope(
            [layers.conv2d],
            kernel_size=[4, 4],
            stride=2,
            padding='valid',
            activation_fn=activation_fn):

        # No normalization on the input layer.
        net = layers.conv2d(
            padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')

        end_points['conv0'] = net

        for i in range(1, num_layers - 1):
            net = layers.conv2d(
                padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)
            end_points['conv%d' % i] = net

        # Stride 1 on the last layer.
        net = layers.conv2d(
            padded(net, 'conv%d' % (num_layers - 1)),
            num_filters[-1],
            stride=1,
            scope='conv%d' % (num_layers - 1))
        end_points['conv%d' % (num_layers - 1)] = net

        # 1-dim logits, stride 1, no activation, no normalization.
        logits = layers.conv2d(
            padded(net, 'conv%d' % num_layers),
            1,
            stride=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv%d' % num_layers)
        end_points['logits'] = logits
        end_points['predictions'] = tf.sigmoid(logits)
    return logits, end_points


def cyclegan_upsample(net, num_outputs, stride, method='conv2d_transpose',
                      pad_mode='REFLECT', align_corners=False):
    """Upsamples the given inputs.

    Args:
      net: A Tensor of size [batch_size, height, width, filters].
      num_outputs: The number of output filters.
      stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
        relative to the inputs, of the output dimensions. For example, if kernel
        size is [2, 3], then the output height and width will be twice and three
        times the input size.
      method: The upsampling method: 'nn_upsample_conv', 'bilinear_upsample_conv',
        or 'conv2d_transpose'.
      pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".
      align_corners: option for method, 'bilinear_upsample_conv'. If true, the
        centers of the 4 corner pixels of the input and output tensors are
        aligned, preserving the values at the corner pixels.

    Returns:
      A Tensor which was upsampled using the specified method.

    Raises:
      ValueError: if `method` is not recognized.
    """
    with tf.variable_scope('upconv'):
        net_shape = tf.shape(net)
        height = net_shape[1]
        width = net_shape[2]

        # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a 3x3
        # 'valid' convolution produce an output with the same dimension as the
        # input.
        spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

        if method == 'nn_upsample_conv':
            net = tf.image.resize_nearest_neighbor(
                net, [stride[0] * height, stride[1] * width])
            net = tf.pad(net, spatial_pad_1, pad_mode)
            net = layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        elif method == 'bilinear_upsample_conv':
            net = tf.image.resize_bilinear(
                net, [stride[0] * height, stride[1] * width],
                align_corners=align_corners)
            net = tf.pad(net, spatial_pad_1, pad_mode)
            net = layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        elif method == 'conv2d_transpose':
            # This corrects 1 pixel offset for images with even width and height.
            # conv2d is left aligned and conv2d_transpose is right aligned for even
            # sized images (while doing 'SAME' padding).
            # Note: This doesn't reflect actual model in paper.
            net = layers.conv2d_transpose(
                net, num_outputs, kernel_size=[3, 3], stride=stride, padding='valid')
            net = net[:, 1:, 1:, :]
        else:
            raise ValueError('Unknown method: [%s]' % method)

        return net


def _dynamic_or_static_shape(tensor):
    shape = tf.shape(tensor)
    static_shape = tf.contrib.util.constant_value(shape)
    return static_shape if static_shape is not None else shape


def cyclegan_generator_resnet(images,
                              arg_scope_fn=cyclegan_arg_scope,
                              num_resnet_blocks=6,
                              num_filters=64,
                              upsample_fn=cyclegan_upsample,
                              kernel_size=3,
                              tanh_linear_slope=0.0,
                              is_training=False):
    """Defines the cyclegan resnet network architecture.

    As closely as possible following
    https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L232

    FYI: This network requires input height and width to be divisible by 4 in
    order to generate an output with shape equal to input shape. Assertions will
    catch this if input dimensions are known at graph construction time, but
    there's no protection if unknown at graph construction time (you'll see an
    error).

    Args:
      images: Input image tensor of shape [batch_size, h, w, 3].
      arg_scope_fn: Function to create the global arg_scope for the network.
      num_resnet_blocks: Number of ResNet blocks in the middle of the generator.
      num_filters: Number of filters of the first hidden layer.
      upsample_fn: Upsampling function for the decoder part of the generator.
      kernel_size: Size w or list/tuple [h, w] of the filter kernels for all inner
        layers.
      tanh_linear_slope: Slope of the linear function to add to the tanh over the
        logits.
      is_training: Whether the network is created in training mode or inference
        only mode. Not actually needed, just for compliance with other generator
        network functions.

    Returns:
      A `Tensor` representing the model output and a dictionary of model end
        points.

    Raises:
      ValueError: If the input height or width is known at graph construction time
        and not a multiple of 4.
    """
    # Neither dropout nor batch norm -> dont need is_training
    del is_training

    end_points = {}

    input_size = images.shape.as_list()
    height, width = input_size[1], input_size[2]
    if height and height % 4 != 0:
        raise ValueError('The input height must be a multiple of 4.')
    if width and width % 4 != 0:
        raise ValueError('The input width must be a multiple of 4.')
    num_outputs = input_size[3]

    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size, kernel_size]

    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height // 2
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width // 2
    paddings = np.array(
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        dtype=np.int32)
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

    with tf.contrib.framework.arg_scope(arg_scope_fn()):

        ###########
        # Encoder #
        ###########
        with tf.variable_scope('input'):
            # 7x7 input stage
            net = tf.pad(images, spatial_pad_3, 'REFLECT')
            net = layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
            end_points['encoder_0'] = net

        with tf.variable_scope('encoder'):
            with tf.contrib.framework.arg_scope(
                    [layers.conv2d],
                    kernel_size=kernel_size,
                    stride=2,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                net = tf.pad(net, paddings, 'REFLECT')
                net = layers.conv2d(net, num_filters * 2)
                end_points['encoder_1'] = net
                net = tf.pad(net, paddings, 'REFLECT')
                net = layers.conv2d(net, num_filters * 4)
                end_points['encoder_2'] = net

        ###################
        # Residual Blocks #
        ###################
        with tf.variable_scope('residual_blocks'):
            with tf.contrib.framework.arg_scope(
                    [layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu,
                    padding='VALID'):
                for block_id in xrange(num_resnet_blocks):
                    with tf.variable_scope('block_{}'.format(block_id)):
                        res_net = tf.pad(net, paddings, 'REFLECT')
                        res_net = layers.conv2d(res_net, num_filters * 4)
                        res_net = tf.pad(res_net, paddings, 'REFLECT')
                        res_net = layers.conv2d(res_net, num_filters * 4,
                                                activation_fn=None)
                        net += res_net

                        end_points['resnet_block_%d' % block_id] = net

        ###########
        # Decoder #
        ###########
        with tf.variable_scope('decoder'):
            with tf.contrib.framework.arg_scope(
                    [layers.conv2d],
                    kernel_size=kernel_size,
                    stride=1,
                    activation_fn=tf.nn.relu):
                with tf.variable_scope('decoder1'):
                    net = upsample_fn(net, num_outputs=num_filters * 2, stride=[2, 2])
                end_points['decoder1'] = net

                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
                end_points['decoder2'] = net

        with tf.variable_scope('output'):
            net = tf.pad(net, spatial_pad_3, 'REFLECT')
            logits = layers.conv2d(
                net,
                num_outputs, [7, 7],
                activation_fn=None,
                normalizer_fn=None,
                padding='valid')
            logits = tf.reshape(logits, _dynamic_or_static_shape(images))

            end_points['logits'] = logits
            end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope

    return end_points['predictions'], end_points


def generator(input_images):
    """Thin wrapper around CycleGAN generator to conform to the TFGAN API.

    Args:
      input_images: A batch of images to translate. Images should be normalized
        already. Shape is [batch, height, width, channels].

    Returns:
      Returns generated image batch.

    Raises:
      ValueError: If shape of last dimension (channels) is not defined.
    """
    input_images.shape.assert_has_rank(4)
    input_size = input_images.shape.as_list()
    channels = input_size[-1]
    if channels is None:
        raise ValueError(
            'Last dimension shape must be known but is None: %s' % input_size)
    with tf.contrib.framework.arg_scope(cyclegan_arg_scope()):
        output_images, _ = cyclegan_generator_resnet(input_images)
    return output_images


def discriminator(image_batch, unused_conditioning=None):
    """A thin wrapper around the Pix2Pix discriminator to conform to TFGAN API."""
    with tf.contrib.framework.arg_scope(pix2pix_arg_scope()):
        logits_4d, _ = pix2pix_discriminator(
            image_batch, num_filters=[64, 128, 256, 512])
        logits_4d.shape.assert_has_rank(4)
    # Output of logits is 4D. Reshape to 2D, for TFGAN.
    logits_2d = tf.contrib.layers.flatten(logits_4d)

    return logits_2d
