"""
"""
import functools

import tensorflow as tf


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

    normalize = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    activate = tf.nn.relu

    tensors = images

    for index, filters in enumerate([8, 16, 32, 64]):
        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1 if index == 0 else 2,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='encode_{}_0'.format(index))

        tensors = normalize(tensors)

        tensors = activate(tensors)

        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='encode_{}_1'.format(index))

        tensors = normalize(tensors)

        tensors = activate(tensors)

    embeddings = tensors

    for index, filters in enumerate([64, 32, 16, 8]):
        tensors = tf.layers.conv2d_transpose(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1 if index == 0 else 2,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='decode_{}_0'.format(index))

        tensors = normalize(tensors)

        tensors = activate(tensors)

        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='decode_{}_1'.format(index))

        tensors = normalize(tensors)

        tensors = activate(tensors)

    tensors = tf.layers.conv2d(
        tensors,
        filters=3,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        name='decode_{}_e'.format(index))

    tensors = tf.nn.tanh(tensors + images)

    logits = tf.identity(tensors, name='logits')

    # NOTE: build a simplier model without traning op
    if labels is None:
        return {
            'images': images,
            'logits': logits,
            'training': training,
        }

    loss_reconstruction = tf.losses.mean_squared_error(
        tensors, labels, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    loss = loss_reconstruction

    return {
        'images': images,
        'labels': labels,
        'logits': logits,
        'loss': loss,
        'training': training,
    }
