"""
"""
import functools

import tensorflow as tf


def block(
        tensors,
        filters,
        kernel_size,
        fn_activate,
        fn_normalize,
        initializer,
        name):
    """
    """
    with tf.variable_scope(name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=fn_activate,
            use_bias=True,
            kernel_initializer=initializer,
            name='c0')

    return tensors


def build_model(
        images_1x, images_2x, images_4x, labels_1x, labels_2x, labels_4x):
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

#   initializer = tf.truncated_normal_initializer(stddev=0.02)

    initializer = tf.contrib.layers.xavier_initializer_conv2d()

    fn_normalize = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    fn_activate = tf.nn.leaky_relu

    fn_block = functools.partial(
        block,
        filters=64,
        kernel_size=5,
        fn_activate=fn_activate,
        fn_normalize=fn_normalize,
        initializer=initializer)

    # NOTE: 1x
    features_1x = []

    tensors = fn_block(images_1x, name='1x-init')

    for i in range(6):
        tensors = fn_block(tensors, name='1x-{}'.format(i))

        features_1x.append(tensors)

    tensors = tf.concat(features_1x, -1)

    logits_1x = images_1x + fn_block(tensors, filters=3, name='1x-res')

    # NOTE: 2x
    tensors_x = fn_block(images_2x, name='2x-init-x')

    tensors_y = tf.layers.conv2d_transpose(
        tensors,
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        data_format='channels_last',
        activation=fn_activate,
        use_bias=True,
        kernel_initializer=initializer,
        name='2x-init-y')

    tensors = tf.concat([tensors_x, tensors_y], -1)

    features_2x = []

    for i in range(6):
        tensors = fn_block(tensors, name='2x-{}'.format(i))

        features_2x.append(tensors)

    tensors = tf.concat(features_2x, -1)

    logits_2x = images_2x + fn_block(tensors, filters=3, name='2x-res')

    # NOTE: 4x
    tensors_x = fn_block(images_4x, name='4x-init-x')

    tensors_y = tf.layers.conv2d_transpose(
        tensors,
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        data_format='channels_last',
        activation=fn_activate,
        use_bias=True,
        kernel_initializer=initializer,
        name='4x-init-y')

    tensors = tf.concat([tensors_x, tensors_y], -1)

    features_4x = []

    for i in range(6):
        tensors = fn_block(tensors, name='4x-{}'.format(i))

        features_4x.append(tensors)

    tensors = tf.concat(features_4x, -1)

    logits_4x = images_4x + fn_block(tensors, filters=3, name='4x-res')

    # NOTE:
    loss_1x = tf.losses.mean_squared_error(
        logits_1x, labels_1x, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    loss_2x = tf.losses.mean_squared_error(
        logits_2x, labels_2x, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    loss_4x = tf.losses.mean_squared_error(
        logits_4x, labels_4x, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    loss = 0.04 * loss_1x + 0.16 * loss_2x + 0.8 * loss_4x

    return {
        'images_1x': images_1x,
        'images_2x': images_2x,
        'images_4x': images_4x,
        'labels_1x': labels_1x,
        'labels_2x': labels_2x,
        'labels_4x': labels_4x,
        'logits_2x': logits_2x,
        'logits_4x': logits_4x,
        'loss': loss,
        'training': training,
    }
