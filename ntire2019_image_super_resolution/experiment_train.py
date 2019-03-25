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

    # NOTE: build training dataset generator
    training_dataset = dataset.TrainingDataset(
        source_dir_path=FLAGS.training_image_source_dir_path,
        target_dir_path=FLAGS.training_image_target_dir_path,
        batch_size=FLAGS.training_batch_size,
        image_size=FLAGS.training_image_size,
        pool_size=5,
        refresh_rate=8)

    # NOTE: build validation image
    test_image_names = tf.gfile.ListDirectory(
        FLAGS.validation_image_source_dir_path)

    source_image_path = os.path.join(
        FLAGS.validation_image_source_dir_path, test_image_names[0])
    target_image_path = os.path.join(
        FLAGS.validation_image_target_dir_path, test_image_names[0])

    validation_image = dataset.TestImage(
        source_image_path=source_image_path,
        target_image_path=target_image_path,
        patch_size=FLAGS.testing_patch_size,
        cover_size=FLAGS.testing_overlapping_size)

    # NOTE: structure of test dataset is different from training & validation
    #       sets. do the ETL in test phase.

    return {
        'training_dataset': training_dataset,
        'validation_image': validation_image,
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

    FLAGS = tf.app.flags.FLAGS

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


def search_learning_rate(session, experiment):
    """
    arXiv:1506.01186, Cyclical Learning Rates for Training Neural Networks

    3.3 how can one estimate reasonable minimum and maximum boundary values
    """
    FLAGS = tf.app.flags.FLAGS

    data, model = experiment['data'], experiment['model']

    learning_rate = FLAGS.training_lrs_initial_rate

    # NOTE: feeds for computing gradients
    forward_feeds = {
        model['training']: True,
#       model['loss_weight']: 0.9,
    }

    while True:
        # NOTE: feeds for backpropagation (larger batch aggregation)
        backward_feeds = {
            model['learning_rate']: learning_rate,
        }

        losses = 0.0

        all_gradients = []

        # NOTE: compute gradients on nano batches
        for i in range(FLAGS.training_batch_size_multiplier):
            # NOTE: training data batch
            batch = data['training_dataset'].next_batch()

            forward_feeds[model['images_1x']] = batch['source_patches_1x']
            forward_feeds[model['images_2x']] = batch['source_patches_2x']
            forward_feeds[model['images_4x']] = batch['source_patches_4x']
            forward_feeds[model['labels_1x']] = batch['target_patches_1x']
            forward_feeds[model['labels_2x']] = batch['target_patches_2x']
            forward_feeds[model['labels_4x']] = batch['target_patches_4x']

            loss, gradients = session.run(
                [model['loss'], model['gradients_result']],
                feed_dict=forward_feeds)

            losses += loss

            all_gradients.append(gradients)

            for i, gradients_source in enumerate(model['gradients_source']):
                gradients = np.stack([g[i] for g in all_gradients], axis=0)

                backward_feeds[gradients_source] = np.mean(gradients, axis=0)

        # NOTE: apply aggregated gradients to gradient descent
        session.run(model['optimizer'], feed_dict=backward_feeds)

        loss = losses / FLAGS.training_batch_size_multiplier

        # NOTE: training log
        step = session.run(model['step'])

        summary = tf.Summary(
            value=[tf.Summary.Value(tag='lrs_loss', simple_value=loss)])

        experiment['scribe'].add_summary(summary, step)

        # NOTE: end of searching
        if step >= FLAGS.training_lrs_stop_step:
            break

        # NOTE: in each step, slightly incereasing the learning rate
        learning_rate += FLAGS.training_lrs_increasing_rate


def train(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data, model = experiment['data'], experiment['model']

    # NOTE: feeds for training
    losses = 0.0

    all_gradients = []

    # NOTE: loss weight
    step = session.run(model['step'])
#   step_begin = FLAGS.training_loss_weight_shifting_begin
#   step_end = FLAGS.training_loss_weight_shifting_end
#   weight = np.clip((step - step_begin) / (step_end - step_begin), 0.0, 1.0)
#   weight = 0.9

    feeds = {
        model['training']: True,
#       model['loss_weight']: weight,
    }

    # NOTE: compute gradients on nano batches
    batch_size_multiplier = FLAGS.training_batch_size_multiplier

    for i in range(batch_size_multiplier):
        # NOTE: training data batch
        batch = data['training_dataset'].next_batch()

        feeds[model['images_1x']] = batch['source_patches_1x']
        feeds[model['images_2x']] = batch['source_patches_2x']
        feeds[model['images_4x']] = batch['source_patches_4x']
        feeds[model['labels_1x']] = batch['target_patches_1x']
        feeds[model['labels_2x']] = batch['target_patches_2x']
        feeds[model['labels_4x']] = batch['target_patches_4x']

        loss, gradients = session.run(
            [model['loss'], model['gradients_result']], feed_dict=feeds)

        losses += loss

        all_gradients.append(gradients)

    # NOTE: aggregate & apply gradients
    step_begin = FLAGS.training_learning_rate_decay_begin
    step_cycle = FLAGS.training_learning_rate_decay_cycle
    decay_rate = FLAGS.training_learning_rate_decay_rate

    decay_rate = decay_rate ** max(0, (step - step_begin) // step_cycle)

    feeds = {
        model['learning_rate']: FLAGS.training_learning_rate * decay_rate,
    }

    for i, gradients_source in enumerate(model['gradients_source']):
        gradients = np.stack([g[i] for g in all_gradients], axis=0)

        feeds[gradients_source] = np.mean(gradients, axis=0)

    # NOTE: apply aggregated gradients to gradient descent
    session.run(model['optimizer'], feed_dict=feeds)

    loss = losses / batch_size_multiplier

    step = session.run(model['step'])

    # NOTE: training log
    summary = tf.Summary(
        value=[tf.Summary.Value(tag='train_loss', simple_value=loss)])

    experiment['scribe'].add_summary(summary, step)

    if step % 1000 == 0:
        ts = datetime.datetime.now().isoformat()
        tf.logging.info('{} - loss[{}]: {}'.format(ts, step, loss))


def validate(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data, model = experiment['data'], experiment['model']

    step = session.run(model['step'])

    if step % FLAGS.validation_cycle != 0:
        return

    # NOTE: collect results of validation
    validation_image = data['validation_image']

    feeds = {
        model['training']: False,
    }

    for begin in range(0, validation_image.num_patches(), 32):
        end = min(validation_image.num_patches(), begin + 32)

        patches_1x, patches_2x, patches_4x = \
            validation_image.get_source_patches(begin, end)

        feeds[model['images_1x']] = patches_1x
        feeds[model['images_2x']] = patches_2x
        feeds[model['images_4x']] = patches_4x

        logits = session.run(model['logits_4x'], feed_dict=feeds)

        validation_image.set_result_patches(logits, begin)

    # NOTE: image summary
    image = validation_image.sample()

    image_stream = io.BytesIO()

    imageio.imwrite(image_stream, image, 'png')

    image = tf.Summary.Image(
        encoded_image_string=image_stream.getvalue(),
        height=image.shape[0],
        width=image.shape[1])

    summary_image = tf.Summary.Value(tag='validation_image', image=image)

    # NOTE:
    psnr = validation_image.psnr()

    summary_psnr = tf.Summary.Value(tag='validation_psnr', simple_value=psnr)

    # NOTE: make image summary and add it
    summary = tf.Summary(value=[summary_image, summary_psnr])

    experiment['scribe'].add_summary(summary, step)


def test(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data, model = experiment['data'], experiment['model']

    step = session.run(model['step'])

    if step % FLAGS.testing_cycle != 0 and step != FLAGS.training_stop_step:
        return

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


def save(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = experiment['model']

    step = session.run(model['step'])

    # NOTE:
    if step % FLAGS.save_cycle == 0:
        target_ckpt_path = os.path.join(FLAGS.checkpoint_path, 'model.ckpt')

        tf.train.Saver().save(session, target_ckpt_path, global_step=step)


def train_validate_test(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = experiment['model']

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    if source_ckpt_path is None:
        session.run(tf.global_variables_initializer())
    else:
        tf.train.Saver().restore(session, source_ckpt_path)

    step = session.run(model['step'])

    # NOTE: exclude log which does not happend yet :)
    experiment['scribe'].add_session_log(
        tf.SessionLog(status=tf.SessionLog.START), global_step=step)

    while session.run(model['step']) < FLAGS.training_stop_step:
        train(session, experiment)

        save(session, experiment)

        validate(session, experiment)

        test(session, experiment)


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

    # NOTE: build file writer to keep log
    experiment['scribe'] = tf.summary.FileWriter(FLAGS.summary_path)

    with tf.Session() as session:
        # NOTE: initialize all variables
        session.run(tf.global_variables_initializer())

        if FLAGS.training_lrs_increasing_rate > 0.0:
            search_learning_rate(session, experiment)
        else:
            train_validate_test(session, experiment)


if __name__ == '__main__':

    tf.app.flags.DEFINE_string('checkpoint_path', None, '')
    tf.app.flags.DEFINE_string('summary_path', None, '')

    # NOTE: path to a dir contains multiple TFRecord files
    tf.app.flags.DEFINE_string('training_image_source_dir_path', None, '')

    tf.app.flags.DEFINE_string('training_image_target_dir_path', None, '')

    tf.app.flags.DEFINE_integer('training_batch_size', 0, '')

    tf.app.flags.DEFINE_integer('training_batch_size_multiplier', 1, '')

    tf.app.flags.DEFINE_integer('training_image_size', 0, '')

    tf.app.flags.DEFINE_integer('training_stop_step', 0, '')

    tf.app.flags.DEFINE_integer('training_loss_weight_shifting_begin', 0, '')

    tf.app.flags.DEFINE_integer('training_loss_weight_shifting_end', 0, '')

    tf.app.flags.DEFINE_integer('training_learning_rate_decay_begin', 0, '')

    tf.app.flags.DEFINE_integer('training_learning_rate_decay_cycle', 0, '')

    tf.app.flags.DEFINE_float('training_learning_rate_decay_rate', 0, '')

    tf.app.flags.DEFINE_float('training_learning_rate', 0.0001, '')

    # NOTE: search proper learning rate base on training parameters (e.g.
    #       batch size & multiplier)
    tf.app.flags.DEFINE_integer(
        'training_lrs_stop_step',
        0,
        'stop learning rate search at the specified step')

    tf.app.flags.DEFINE_float(
        'training_lrs_initial_rate',
        0.0,
        'search learning rate start at the specified value')

    tf.app.flags.DEFINE_float(
        'training_lrs_increasing_rate',
        0.0001,
        'increase learning rate with the specified value in each step')

    # NOTE:
    tf.app.flags.DEFINE_string('validation_image_source_dir_path', None, '')

    tf.app.flags.DEFINE_string('validation_image_target_dir_path', None, '')

    tf.app.flags.DEFINE_integer('validation_cycle', 1000, '')

    tf.app.flags.DEFINE_integer('validation_batch_size', 0, '')

    tf.app.flags.DEFINE_integer('validation_batch_size_multiplier', 1, '')

    tf.app.flags.DEFINE_integer('validation_image_size', 0, '')

    # NOTE:
    tf.app.flags.DEFINE_string('testing_image_source_dir_path', None, '')

    tf.app.flags.DEFINE_integer('testing_cycle', 1000, '')

    tf.app.flags.DEFINE_integer('testing_batch_size', 0, '')

    tf.app.flags.DEFINE_integer('testing_patch_size', 16, '')

    tf.app.flags.DEFINE_integer('testing_overlapping_size', 0, '')

    tf.app.flags.DEFINE_string('testing_result_path', None, '')

    # NOTE:
    tf.app.flags.DEFINE_integer('save_cycle', 1000, '')

    # NOTE:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
