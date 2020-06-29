#!/usr/bin/env python
"""
    Script which runs the training procedure
"""
import os
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset


# Shortcut aliases
layers = tf.keras.layers
models = tf.keras.models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_integer('epochs', 5, 'Number of training epochs')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate of ADAM optimizer')
flags.DEFINE_string('output_path',
                    os.path.join(REPO_DIR, 'saved_model'),
                    'Path to trained model')


def build_dataset(batch_size: int) -> Tuple[PrefetchDataset, PrefetchDataset]:
    """ Load train/test dataset for MNIST """
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    ds_train = ds_train.map(normalize,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test


def normalize(image: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    """ Normalize images based on 255 color intensity """
    return tf.cast(image, tf.float32) / 255., label


def build_model(learning_rate: float) -> models.Sequential:
    """ Build tf.keras model to train """
    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1), name='inputs'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(10, activation='softmax', name='outputs')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy'],
    )
    return model


def main(_):
    """ Main training execution """
    ds_train, ds_test = build_dataset(FLAGS.batch_size)
    model = build_model(FLAGS.learning_rate)

    model.fit(
        ds_train,
        epochs=FLAGS.epochs,
        validation_data=ds_test,
    )
    model.save(FLAGS.output_path)


if __name__ == '__main__':
    app.run(main)
