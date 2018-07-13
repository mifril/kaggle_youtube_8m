import os
import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import TRAIN_PATH, TEST_PATH, VAL_PATH, N_CLASSES


class Dataset:
    def __init__(self, args, mode):
        self._mode = mode
        self._batch_size = args.batch_size
        self._use_tf_records = args.use_tf_records

        if self._mode == 'train':
            self._in_path = TRAIN_PATH
            self._len = 4421199
        elif self._mode == 'val':
            self._in_path = VAL_PATH
            self._len = 580076
        elif self._mode == 'test':
            self._in_path = TEST_PATH
            self._len = 1133323
        else:
            raise ValueError('Invalid mode: {}'.format(self._mode))

        if self._use_tf_records:
            self._files = sorted(glob.glob(os.path.join(self._in_path, '*.tfrecord')))
        else:
            self._files = sorted(glob.glob(os.path.join(self._in_path, '*.npz')))

        # if self._mode != 'test':
        #     self._ids = self._get_ids()
        #     self._len = len(self._ids)

        print('Mode: {}. Total files: {}'.format(self._mode, len(self._files)))

    def __len__(self):
        return self._len

    def generator(self):
        if self._use_tf_records:
            return self._tf_iterator()
        else:
            return self._npz_iterator()

    def _get_ids(self):
        ids = []
        for fn in tqdm(self._files, 'Count ids'):
            for example in tf.python_io.tf_record_iterator(fn):
                tf_example = tf.train.Example.FromString(example)
                ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        return ids

    # based on https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb
    def _tf_iterator(self):
        ids, audio, rgb, labels = [], [], [], []
        for fn in self._files:
            for example in tf.python_io.tf_record_iterator(fn):
                tf_example = tf.train.Example.FromString(example)

                ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
                rgb.append(np.array(tf_example.features.feature['mean_rgb'].float_list.value))
                audio.append(np.array(tf_example.features.feature['mean_audio'].float_list.value))

                example_labels = np.array(tf_example.features.feature['labels'].int64_list.value)
                example_labels_ohe = np.zeros(N_CLASSES).astype(np.int8)
                for label in example_labels:
                    example_labels_ohe[label] = 1
                labels.append(example_labels_ohe)

                if len(ids) >= self._batch_size:
                    yield np.array(ids), {'audio': np.array(audio), 'rgb': np.array(rgb)}, np.array(labels)
                    ids, audio, rgb, labels = [], [], [], []
        if len(ids) >= 0:
            yield np.array(ids), {'audio': np.array(audio), 'rgb': np.array(rgb)}, np.array(labels)

    def _npz_iterator(self):
        for fn in self._files:
            loaded = np.load(fn)
            labels = loaded['labels']
            labels_ohe = np.zeros((labels.shape[0], N_CLASSES)).astype(np.uint8)
            for i in range(labels.shape[0]):
                labels_ohe[i, labels[i]] = 1
            yield loaded['ids'], {'audio': loaded['audio'].astype(np.float32), 'rgb': loaded['rgb'].astype(np.float32)}, labels_ohe
