"""
"""
import os

import imageio
import numpy as np
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

            source_image = imageio.imread(source_path).astype(np.float32)
            target_image = imageio.imread(target_path).astype(np.float32)

            source_image = source_image / 127.5 - 1.0
            target_image = target_image / 127.5 - 1.0

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
        shape = (self._batch_size, self._image_size, self._image_size, 3)

        source_images = np.zeros(shape, dtype=np.float32)
        target_images = np.zeros(shape, dtype=np.float32)

        for i in range(self._batch_size):
            j = np.random.randint(len(self._data_pool))

            source_image = self._data_pool[j]['source_image']
            target_image = self._data_pool[j]['target_image']

            s = self._image_size

            h, w, _ = source_image.shape

            # NOTE: augmentation, random crop
            y = np.random.randint(h - s)
            x = np.random.randint(w - s)

            source_patch = source_image[y:y+s, x:x+s]
            target_patch = target_image[y:y+s, x:x+s]

            # NOTE: augmentation, flip vertically
            if np.random.choice([True, False]):
                source_patch = source_patch[::-1, ::]
                target_patch = target_patch[::-1, ::]


            # NOTE: augmentation, flip horizontally
            if np.random.choice([True, False]):
                source_patch = source_patch[::, ::-1]
                target_patch = target_patch[::, ::-1]

            # NOTE: augmentation, transpose x-y axis
            if np.random.choice([True, False]):
                source_patch = np.transpose(source_patch, [1, 0, 2])
                target_patch = np.transpose(target_patch, [1, 0, 2])

            source_images[i] = source_patch
            target_images[i] = target_patch

        return source_images, target_images


class TestImage:
    """
    """
    def __init__(self, **kwargs):
        """
        source_image_path
        patch_size
        overlap_size
        """
        self._patch_size = kwargs.get('patch_size', 64)
        self._overlap_size = kwargs.get('overlap_size', 16)

        source_image_path = kwargs.get('source_image_path', None)

        # NOTE: sanity check, patch_size must be greater then 2 x overlap_size
        if self._patch_size <= 2 * self._overlap_size:
            raise Exception('invalid patch_size & overlap_size')

        # NOTE: sanity check, the image must exist
        if source_image_path is None or not tf.gfile.Exists(source_image_path):
            raise Exception('source_image_path must exist')

        image = imageio.imread(source_image_path).astype(np.float32)
        image = image / 127.5 - 1.0

        self._image_height = image.shape[0]
        self._image_width = image.shape[1]

        h, w, _ = image.shape

        p, o = self._patch_size, self._overlap_size

        z = p - 2 * o

        ch, cw = (h + z - 1) // z, (w + z - 1) // z

        temp_image = np.zeros((ch * z + 2 * o, cw * z + 2 * o, 3), np.float32)

        temp_image[o:o+h, o:o+w] = image

        self._tiles = []

        for y in range(ch):
            for x in range(cw):
                ty, tx = y * z, x * z

                self._tiles.append(temp_image[ty:ty+p, tx:tx+p: ])

        self._tiles = np.stack(self._tiles)

    def num_tiles(self):
        """
        """
        return self._tiles.shape[0]

    def get_low_resolution_tiles(self, begin, end):
        """
        """
        return self._tiles[begin:end]

    def set_super_resolved_tiles(self, tiles, begin):
        """
        """
        self._tiles[begin:begin+tiles.shape[0]] = tiles

    def save_super_resolved_image(self, path):
        """
        """
        h, w = self._image_height, self._image_width

        p, o = self._patch_size, self._overlap_size

        z = p - 2 * o

        ch, cw = (h + z - 1) // z, (w + z - 1) // z

        image = np.zeros((ch * z, cw * z, 3), np.float32)

        i = 0

        for y in range(0, ch * z, z):
            for x in range(0, cw * z, z):
                image[y:y+z, x:x+z] = self._tiles[i, o:o+z, o:o+z]

                i += 1

        image = image[:self._image_height, :self._image_width]

        image = np.clip(image * 127.5 + 127.5, 0.0, 255.0)

        image = image.astype(np.uint8)

        imageio.imwrite(path, image, 'png')

