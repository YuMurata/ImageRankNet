import tensorflow as tf
import numpy as np
from pathlib import Path
from config import IMAGE_SHAPE, tfrecords_dir


def make_example(image1: np.array, image2: np.array, label: int):
    assert image1.shape == image2.shape

    feature = tf.train.Features(feature={
        'image1':
        tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[image1.tobytes()])),
        'image2':
        tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[image2.tobytes()])),
        'label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    })
    return tf.train.Example(features=feature)


def write_tfrecord(filename: str):
    writer = tf.io.TFRecordWriter(filename)
    for _ in range(10):
        white = np.ones(IMAGE_SHAPE) * 255 - \
            np.random.uniform(high=10, size=IMAGE_SHAPE)

        black = np.zeros(IMAGE_SHAPE) + \
            np.random.uniform(high=10, size=IMAGE_SHAPE)

        label = 0

        ex = make_example(white.astype(np.int8), black.astype(np.int8), label)
        writer.write(ex.SerializeToString())

    writer.close()


if __name__ == "__main__":
    write_tfrecord(str(tfrecords_dir / 'train.tfrecord'))
    write_tfrecord(str(tfrecords_dir / 'validation.tfrecord'))
