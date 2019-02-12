"""
"""
import functools

import tensorflow as tf


def block(tensors, initializer, fn_normalize, fn_activate, name):
    """
    """
    num_channels_in = tensors.shape[-1]

    with tf.variable_scope(name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=num_channels_in*2,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='conv_3x3')

        tensors = fn_activate(tensors)

        tensors = fn_normalize(tensors)

    return tensors


def build_model(images, labels):
    """
    images:
        feature batch for supervised learning. the noisy images in this
        denoising task.
    labels:
        label batch for supervised learning. the ground truth images in this
        denoising task.
    """
    # NOTE: a tensor to control ops such as batch normalization, which should
    #       stop updating its internal state during validation & testing.
    training = tf.placeholder(shape=[], dtype=tf.bool)

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    fn_normalize = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    fn_activate = tf.nn.relu

    tensors = images

    scale_1 = block(tensors, initializer, fn_normalize, fn_activate, 's1')
    scale_2 = block(scale_1, initializer, fn_normalize, fn_activate, 's2')
    scale_3 = block(scale_2, initializer, fn_normalize, fn_activate, 's3')
    scale_4 = block(scale_3, initializer, fn_normalize, fn_activate, 's4')

    tensors = tf.concat([scale_1, scale_2, scale_3, scale_4], -1)

    tensors = tf.layers.conv2d(
        tensors,
        filters=3,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        name='1x1')

    logits = tf.nn.tanh(tensors + images, name='logits')

    # NOTE: build a simplier model without traning op
    if labels is None:
        return {
            'images': images,
            'logits': logits,
            'training': training,
        }

    loss_reconstruction = tf.losses.mean_squared_error(
        logits, labels, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    return {
        'images': images,
        'labels': labels,
        'logits': logits,
        'loss': loss_reconstruction,
        'training': training,
    }
