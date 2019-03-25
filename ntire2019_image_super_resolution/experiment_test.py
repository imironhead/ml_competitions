"""
"""
import datetime
import io
import os
import time

import imageio
import numpy as np
import tensorflow as tf
import zipfile

import dataset
import model_perceptual


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: placeholders
    source_images_1x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='source_images_1x')
    source_images_2x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='source_images_2x')
    source_images_4x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='source_images_4x')
    target_images_1x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='target_images_1x')
    target_images_2x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='target_images_2x')
    target_images_4x_placeholder = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32, name='target_images_4x')

    return {
        'source_images_1x_placeholder': source_images_1x_placeholder,
        'source_images_2x_placeholder': source_images_2x_placeholder,
        'source_images_4x_placeholder': source_images_4x_placeholder,
        'target_images_1x_placeholder': target_images_1x_placeholder,
        'target_images_2x_placeholder': target_images_2x_placeholder,
        'target_images_4x_placeholder': target_images_4x_placeholder,
    }


def build_model(data):
    """
    """
    # NOTE: helper function to create a placeholder for a variable
    def placeholder(g):
        return tf.placeholder(shape=g.shape, dtype=g.dtype)

    # NOTE: create the model
    model = model_perceptual.build_model(
        data['source_images_1x_placeholder'],
        data['source_images_2x_placeholder'],
        data['source_images_4x_placeholder'],
        data['target_images_1x_placeholder'],
        data['target_images_2x_placeholder'],
        data['target_images_4x_placeholder'])

    # NOTE: enhance the model to train with larger batch and less GPU memory.
    #       to do so, we have to extract the pipeline of gradient descent so we
    #       can integrate gradients from multiple small batches in one step.

    # NOTE: build optimizer to trigger gradient descent manually
    model['learning_rate'] = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate=model['learning_rate'])

    # NOTE: force ops being updated before computing gradients
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        gradients_and_vars = optimizer.compute_gradients(model['loss'])

    # NOTE: an operator to collect computed gradients
    model['gradients_result'] = [g for g, v in gradients_and_vars]

    gradients_and_vars = [(placeholder(g), v) for g, v in gradients_and_vars]

    # NOTE: a placeholder for feeding manipulated gradients
    model['gradients_source'] = [g for g, v in gradients_and_vars]

    # NOTE: an operator to apply newly computed gradients
    model['step'] = tf.train.get_or_create_global_step()

    model['optimizer'] = optimizer.apply_gradients(
        gradients_and_vars, global_step=model['step'])

    return model


def test(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data, model = experiment['data'], experiment['model']

    step = session.run(model['step'])

    # NOTE: do test
    feeds = {
        model['training']: False,
    }

    time_cost = 0.0
    time_book = 0.0

    source_dir_path = FLAGS.testing_image_source_dir_path

    source_image_names = tf.gfile.ListDirectory(source_dir_path)

    testing_result_path = FLAGS.testing_result_path.replace(
        '.zip', '_{}.zip'.format(step))

    with zipfile.ZipFile(testing_result_path, 'w') as zipped_results:
        for name in source_image_names:
            print('testing: {}'.format(name))

            test_image = dataset.TestImage(
                source_image_path=os.path.join(source_dir_path, name),
                target_image_path=None,
                patch_size=FLAGS.testing_patch_size,
                cover_size=FLAGS.testing_overlapping_size)

            time_book = time.time()

            for begin in range(0, test_image.num_patches(), 32):
                end = min(test_image.num_patches(), begin + 32)

                patches_1x, patches_2x, patches_4x = \
                    test_image.get_source_patches(begin, end)

                feeds[model['images_1x']] = patches_1x
                feeds[model['images_2x']] = patches_2x
                feeds[model['images_4x']] = patches_4x

                logits = session.run(model['logits_4x'], feed_dict=feeds)

                test_image.set_result_patches(logits, begin)

            time_cost += time.time() - time_book

            image_stream = io.BytesIO()

            test_image.save_result_image(image_stream)

            zipped_results.writestr(name, image_stream.getvalue())

        # NOTE: runtime per mega pixel
        runtime_per_image = time_cost / len(source_image_names)

        # NOTE:
        readme_stream = io.StringIO()

        s1 = 'runtime per image [s] : {}\n'.format(runtime_per_image)
        s2 = 'CPU[1] / GPU[0] : 0\n'
        s3 = 'Extra Data [1] / No Extra Data [0] : 0\n'
        s4 = 'Other: --\n'

        readme_stream.write(s1)
        readme_stream.write(s2)
        readme_stream.write(s3)
        readme_stream.write(s4)

        zipped_results.writestr('readme.txt', readme_stream.getvalue())


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: build experiment which keep a model & dataset
    experiment = {}

    # NOTE: build dataset
    experiment['data'] = build_dataset()

    # NOTE: build model
    experiment['model'] = build_model(experiment['data'])

    with tf.Session() as session:
        tf.train.Saver().restore(session, FLAGS.checkpoint_path)

        test(session, experiment)


if __name__ == '__main__':

    tf.app.flags.DEFINE_string('checkpoint_path', None, '')

    # NOTE:
    tf.app.flags.DEFINE_string('testing_image_source_dir_path', None, '')

    tf.app.flags.DEFINE_integer('testing_patch_size', 16, '')

    tf.app.flags.DEFINE_integer('testing_overlapping_size', 0, '')

    tf.app.flags.DEFINE_string('testing_result_path', None, '')

    tf.app.run()
