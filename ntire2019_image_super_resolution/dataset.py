"""
"""
import itertools
import os

import imageio
import numpy as np
import scipy.misc
import tensorflow as tf


class TrainingDataset:
    """
    """
    def __init__(self, **kwargs):
        """
        source_dir_path:
        target_dir_path:
        batch_size:
        image_size:
        pool_size:
        refresh_rate:
        """
        self._batch_size = kwargs.get('batch_size', 16)
        self._image_size = kwargs.get('image_size', 64)
        self._pool_size = kwargs.get('pool_size', 4)
        self._refresh_rate = kwargs.get('refresh_rate', 8)
        self._data_pool = []
        self._path_pool = []
        self._step = 1

        # NOTE: sanity check path of dir that contains images
        source_dir_path = kwargs.get('source_dir_path')
        target_dir_path = kwargs.get('target_dir_path')

        if not source_dir_path:
            raise Exception('source_dir_path for dataset must not be None')

        if not target_dir_path:
            raise Exception('target_dir_path for dataset must not be None')

        if not tf.gfile.Exists(source_dir_path):
            raise Exception('source_dir_path for dataset must exist')

        if not tf.gfile.Exists(target_dir_path):
            raise Exception('target_dir_path for dataset must exist')

        # NOTE: sanity check for pool size & refreshment
        if self._pool_size < 1:
            raise Exception('pool_size must be greater than 0')

        if self._refresh_rate < 1:
            raise Exception('refresh_rate must be greater than 0')

        # NOTE: collect image path
        for name in tf.gfile.ListDirectory(source_dir_path):
            source_path = os.path.join(source_dir_path, name)
            target_path = os.path.join(target_dir_path, name)

            if not tf.gfile.Exists(source_path):
                raise Exception('invalid path: {}'.format(source_path))

            if not tf.gfile.Exists(target_path):
                raise Exception('invalid path: {}'.format(target_path))

            self._path_pool.append({
                'source_path': source_path, 'target_path': target_path})

    def refresh_pool(self):
        """
        """
        # NOTE: remove one set from data_pool every refresh_rate steps
        if self._step % self._refresh_rate == 0 and len(self._data_pool) > 0:
            index = np.random.randint(len(self._data_pool))

            source_path = self._data_pool[index]['source_path']
            target_path = self._data_pool[index]['target_path']

            self._path_pool.append({
                'source_path': source_path, 'target_path': target_path})

            del self._data_pool[index]

        # NOTE: maintain size of data_pool
        while len(self._data_pool) < self._pool_size:
            index = np.random.randint(len(self._path_pool))

            source_path = self._path_pool[index]['source_path']
            target_path = self._path_pool[index]['target_path']

            source_image = imageio.imread(source_path)
            target_image = imageio.imread(target_path)

            if np.random.choice([True, False]):
                h, w, _ = source_image.shape

                h = np.random.randint(h * 990 // 1000, h)
                w = np.random.randint(w * 990 // 1000, w)

                source_image = scipy.misc.imresize(
                    source_image,
                    (h, w),
                    interp='bicubic')

                target_image = scipy.misc.imresize(
                    target_image,
                    (h, w),
                    interp='bicubic')

            source_image = scipy.ndimage.gaussian_filter(
                source_image, sigma=0.5, mode='nearest')

            self._data_pool.append({
                'source_path': source_path,
                'target_path': target_path,
                'source_image': source_image,
                'target_image': target_image})

            del self._path_pool[index]

        self._step += 1

    def next_batch(self):
        """
        """
        self.refresh_pool()

        # NOTE: collect one batch
        batch_size = self._batch_size
        patch_size_4x = self._image_size
        patch_size_2x = patch_size_4x // 2
        patch_size_1x = patch_size_4x // 4

        shape_1x = (batch_size, patch_size_1x, patch_size_1x, 3)
        shape_2x = (batch_size, patch_size_2x, patch_size_2x, 3)
        shape_4x = (batch_size, patch_size_4x, patch_size_4x, 3)

        source_patches_1x = np.zeros(shape_1x, dtype=np.float32)
        source_patches_2x = np.zeros(shape_2x, dtype=np.float32)
        source_patches_4x = np.zeros(shape_4x, dtype=np.float32)
        target_patches_1x = np.zeros(shape_1x, dtype=np.float32)
        target_patches_2x = np.zeros(shape_2x, dtype=np.float32)
        target_patches_4x = np.zeros(shape_4x, dtype=np.float32)

        for i in range(self._batch_size):
            j = np.random.randint(len(self._data_pool))

            source_image = self._data_pool[j]['source_image']
            target_image = self._data_pool[j]['target_image']

            s = patch_size_4x

            h, w, _ = source_image.shape

            # NOTE: augmentation, random crop
            y = np.random.randint(h - s)
            x = np.random.randint(w - s)

            source_patch_4x = source_image[y:y+s, x:x+s]
            target_patch_4x = target_image[y:y+s, x:x+s]

            # NOTE: augmentation, flip vertically
            if np.random.choice([True, False]):
                source_patch_4x = source_patch_4x[::-1, ::]
                target_patch_4x = target_patch_4x[::-1, ::]

            # NOTE: augmentation, flip horizontally
            if np.random.choice([True, False]):
                source_patch_4x = source_patch_4x[::, ::-1]
                target_patch_4x = target_patch_4x[::, ::-1]

            # NOTE: augmentation, transpose x-y axis
            if np.random.choice([True, False]):
                source_patch_4x = np.transpose(source_patch_4x, [1, 0, 2])
                target_patch_4x = np.transpose(target_patch_4x, [1, 0, 2])

            source_patch_1x = scipy.misc.imresize(
                source_patch_4x,
                (patch_size_1x, patch_size_1x),
                interp='bicubic')
            source_patch_2x = scipy.misc.imresize(
                source_patch_4x,
                (patch_size_2x, patch_size_2x),
                interp='bicubic')
            target_patch_1x = scipy.misc.imresize(
                target_patch_4x,
                (patch_size_1x, patch_size_1x),
                interp='bicubic')
            target_patch_2x = scipy.misc.imresize(
                target_patch_4x,
                (patch_size_2x, patch_size_2x),
                interp='bicubic')

            source_patch_1x = source_patch_1x.astype(np.float32) / 127.5 - 1.0
            source_patch_2x = source_patch_2x.astype(np.float32) / 127.5 - 1.0
            source_patch_4x = source_patch_4x.astype(np.float32) / 127.5 - 1.0
            target_patch_1x = target_patch_1x.astype(np.float32) / 127.5 - 1.0
            target_patch_2x = target_patch_2x.astype(np.float32) / 127.5 - 1.0
            target_patch_4x = target_patch_4x.astype(np.float32) / 127.5 - 1.0

            source_patches_1x[i] = source_patch_1x
            source_patches_2x[i] = source_patch_2x
            source_patches_4x[i] = source_patch_4x
            target_patches_1x[i] = target_patch_1x
            target_patches_2x[i] = target_patch_2x
            target_patches_4x[i] = target_patch_4x

        return {
            'source_patches_1x': source_patches_1x,
            'source_patches_2x': source_patches_2x,
            'source_patches_4x': source_patches_4x,
            'target_patches_1x': target_patches_1x,
            'target_patches_2x': target_patches_2x,
            'target_patches_4x': target_patches_4x,}


class TestImage:
    """
    """
    def __init__(self, **kwargs):
        """
        source_image_path
        target_image_path
        patch_size
        cover_size
        """
        source_image_path = kwargs.get('source_image_path', None)
        target_image_path = kwargs.get('target_image_path', None)

        # NOTE: sanity check, the source image must exist
        if source_image_path is None:
            raise Exception('source_image_path must exist')

        if not tf.gfile.Exists(source_image_path):
            raise Exception('source_image_path must exist')

        if target_image_path is None:
            target_image_uint8 = None
        elif not tf.gfile.Exists(target_image_path):
            raise Exception('target_image_path is invalid')
        else:
            self._target_image_uint8 = imageio.imread(target_image_path)

        self._source_image_uint8 = imageio.imread(source_image_path)
        self._source_image_uint8 = scipy.ndimage.gaussian_filter(
            self._source_image_uint8, sigma=0.5, mode='nearest')
        self._result_image_uint8 = np.zeros_like(self._source_image_uint8)

        self._patch_size = patch_size = kwargs.get('patch_size', 64)
        self._cover_size = cover_size = kwargs.get('cover_size', 16)

        # NOTE: sanity check, patch_size must be greater then 2 x cover_size
        if self._patch_size <= 2 * self._cover_size:
            raise Exception('invalid patch_size & cover_size')

        image_h, image_w, _ = self._source_image_uint8.shape

        num_patch_h = (image_h + patch_size - 4 * cover_size - 1) \
            // (patch_size - 2 * cover_size)
        num_patch_w = (image_w + patch_size - 4 * cover_size - 1) \
            // (patch_size - 2 * cover_size)

        self._source_patches_f32 = []

        for y, x in itertools.product(range(num_patch_h), range(num_patch_w)):
            # NOTE: crop a patch
            patch_x = \
                min(x * (patch_size - 2 * cover_size), image_w - patch_size)
            patch_y = \
                min(y * (patch_size - 2 * cover_size), image_h - patch_size)

            patch_4x = self._source_image_uint8[
                patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]

            patch_1x = scipy.misc.imresize(
                patch_4x, (patch_size // 4, patch_size // 4), interp='bicubic')

            patch_1x = patch_1x.astype(np.float32) / 127.5 - 1.0

            patch_2x = scipy.misc.imresize(
                patch_4x, (patch_size // 2, patch_size // 2), interp='bicubic')

            patch_2x = patch_2x.astype(np.float32) / 127.5 - 1.0

            patch_4x = patch_4x.astype(np.float32) / 127.5 - 1.0

            # NOTE: mapping information
            if x == 0 or x + 1 == num_patch_w:
                map_w = patch_size - cover_size
            else:
                map_w = patch_size - cover_size * 2

            if y == 0 or y + 1 == num_patch_h:
                map_h = patch_size - cover_size
            else:
                map_h = patch_size - cover_size * 2

            map_source_x = (0 if x == 0 else cover_size)
            map_source_y = (0 if y == 0 else cover_size)

            if x == 0:
                map_target_x = 0
            else:
                map_target_x = patch_x + cover_size

            if y == 0:
                map_target_y = 0
            else:
                map_target_y = patch_y + cover_size

            self._source_patches_f32.append({
                'patch_1x': patch_1x,
                'patch_2x': patch_2x,
                'patch_4x': patch_4x,
                'map_source_x': map_source_x,
                'map_source_y': map_source_y,
                'map_target_x': map_target_x,
                'map_target_y': map_target_y,
                'map_w': map_w,
                'map_h': map_h})

    def num_patches(self):
        """
        """
        return len(self._source_patches_f32)

    def get_source_patches(self, begin, end):
        """
        """
        patches = self._source_patches_f32[begin:end]

        patches_1x = [patch['patch_1x'] for patch in patches]
        patches_2x = [patch['patch_2x'] for patch in patches]
        patches_4x = [patch['patch_4x'] for patch in patches]

        return np.stack(patches_1x), np.stack(patches_2x), np.stack(patches_4x)

    def set_result_patches(self, patches, begin):
        """
        """
        for i in range(patches.shape[0]):
            info = self._source_patches_f32[begin+i]

            patch = np.clip(patches[i] * 127.5 + 127.5, 0.0, 255.0)
            patch = patch.astype(np.uint8)

            m_s_x = info['map_source_x']
            m_s_y = info['map_source_y']
            m_t_x = info['map_target_x']
            m_t_y = info['map_target_y']
            m_w = info['map_w']
            m_h = info['map_h']

            self._result_image_uint8[m_t_y:m_t_y+m_h, m_t_x:m_t_x+m_w] = \
                patch[m_s_y:m_s_y+m_h, m_s_x:m_s_x+m_w]

    def save_result_image(self, path):
        """
        """
        imageio.imwrite(path, self._result_image_uint8, 'png')

    def psnr(self):
        """
        """
        if self._result_image_uint8 is None:
            return 0.0

        target_image = self._target_image_uint8.astype(np.float32)
        result_image = self._result_image_uint8.astype(np.float32)

        mse = np.mean(np.square(target_image - result_image))

        return 10.0 * np.log10(65025.0 / mse)

    def sample(self, height=128):
        if self._result_image_uint8 is None:
            return None

        ty = np.random.randint(self._source_image_uint8.shape[0] - height)

        return np.concatenate([
            self._source_image_uint8[:height],
            self._result_image_uint8[:height],
            self._target_image_uint8[:height],
            self._source_image_uint8[ty:ty+height],
            self._result_image_uint8[ty:ty+height],
            self._target_image_uint8[ty:ty+height]], axis=0)

