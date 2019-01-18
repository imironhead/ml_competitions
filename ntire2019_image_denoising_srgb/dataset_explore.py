"""
"""
import os
import random

import imageio
import numpy as np
import scipy.io
import tensorflow as tf


def enum_training_image_paired_paths(root_dir_path):
    """
    expect all images reside within root_dir_path like:

    root_dir_path/Data/0001_001_S6_00100_00060_3200_L/0001_GT_SRGB_010.PNG
    root_dir_path/Data/0001_001_S6_00100_00060_3200_L/0001_GT_SRGB_011.PNG
    root_dir_path/Data/0001_001_S6_00100_00060_3200_L/0001_NOISY_SRGB_010.PNG
    root_dir_path/Data/0001_001_S6_00100_00060_3200_L/0001_NOISY_SRGB_011.PNG
    """
    paths = []

    for dir_path, dir_names, file_names in os.walk(root_dir_path):
        for file_name in file_names:
            # NOTE: filter out non png files
            if file_name[-4:].lower() != '.png':
                continue

            # NOTE: filter out non ground truth files
            if '_GT_' not in file_name:
                continue

            # NOTE: build paired names, replace substring to make noisy
            #       counterpart
            file_name_gt = file_name
            file_name_noisy = file_name.replace('_GT_', '_NOISY_')

            # NOTE: collect paired paths
            paths.append({
                'gt': os.path.join(dir_path, file_name_gt),
                'noisy': os.path.join(dir_path, file_name_noisy)})

    return paths


def explore(training_dir_path, validation_mat_path):
    """
    """
    # NOTE: collect all png paths with the specified dir
    #       image_paths is [{'gt': gt, 'noisy': noisy}, ...]
    image_paired_paths = enum_training_image_paired_paths(training_dir_path)

    # NOTE: https://competitions.codalab.org/competitions/21266#participate-get_data
    #       File contents: 320 noisy images in sRGB space with
    #       corresponding ground truth (GT) and metadata.
    if len(image_paired_paths) != 320:
        raise Exception('ntire-19 provides 320 training images')

    # NOTE: not all shape of images are the same
    for paired_paths in image_paired_paths:
        img_gt = imageio.imread(paired_paths['gt'])

        img_noisy = imageio.imread(paired_paths['noisy'])

        if img_gt.shape != img_noisy.shape:
            raise Exception('shape of gt & noisy must be matched')

        # NOTE: noisy = gt + noise, noise is not always positive

    # NOTE: check shape of the validation data
    img = scipy.io.loadmat(validation_mat_path)

    if img['ValidationNoisyBlocksSrgb'].shape != (40, 32, 256, 256, 3):
        raise Exception('shape of validation must be 40x32x256x256x3')


def int64_feature(v):
    """
    create a feature which contains a 64-bits integer
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def raw_feature(v):
    """
    create a feature which contains bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def paired_paths_to_example(paired_paths):
    """
    """
    img_gt = imageio.imread(paired_paths['gt'])
    img_noisy = imageio.imread(paired_paths['noisy'])

    height, width, _ = img_gt.shape

    img_gt = img_gt.flatten().tostring()
    img_noisy = img_noisy.flatten().tostring()

    features = {
        'height': int64_feature(height),
        'width': int64_feature(width),
        'image_gt': raw_feature(img_gt),
        'image_noisy': raw_feature(img_noisy),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def transform(training_dir_path, record_dir_path):
    """
    """
    image_paired_paths = enum_training_image_paired_paths(training_dir_path)

    # NOTE: write gzip
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)

    for index, paired_paths in enumerate(image_paired_paths):
        result_record_path = \
            os.path.join(record_dir_path, '{:0>4}.tfr'.format(index))

        example = paired_paths_to_example(paired_paths)

        with tf.python_io.TFRecordWriter(
                result_record_path, options=options) as writer:
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    import argparse

    # NOTE: explore the raw dataset
    parser = argparse.ArgumentParser(description='explore sidd dataset')

    # NOTE: enable to explore the raw dataset
    parser.add_argument('--explore', action='store_true')

    # NOTE: enable to transform png to TFRecord
    parser.add_argument('--transform', action='store_true')

    # NOTE: path to the root directory of all training images
    parser.add_argument('--training_dir_path', type=str)

    # NOTE: path to the validation data, a MAT
    parser.add_argument('--validation_mat_path', type=str)

    # NOTE: path to the transformed data file in tfrecord format
    parser.add_argument('--record_dir_path', type=str)

    args = parser.parse_args()

    if args.explore:
        explore(args.training_dir_path, args.validation_mat_path)

    if args.transform:
        transform(args.training_dir_path, args.record_dir_path)
