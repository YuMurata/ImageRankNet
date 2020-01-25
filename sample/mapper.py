import tensorflow as tf
import os
import sys
from config import IMAGE_SHAPE

sys.path.append(os.getcwd())
from ImageRankNet import dataset


class SampleMapper(dataset.Mapper):
    def map_example(self, example_proto):
        features = {
            'image1': tf.io.FixedLenFeature((), tf.string,
                                            default_value=""),
            'image2': tf.io.FixedLenFeature((), tf.string,
                                            default_value=""),
            'label': tf.io.FixedLenFeature((), tf.int64,
                                           default_value=0),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        image1_raw = \
            tf.io.decode_raw(parsed_features['image1'], tf.uint8)
        image2_raw =\
            tf.io.decode_raw(parsed_features['image2'], tf.uint8)

        label = tf.cast(parsed_features['label'], tf.int32, name='label')

        float_image1_raw = tf.cast(image1_raw, tf.float32)/255
        float_image2_raw = tf.cast(image2_raw, tf.float32)/255

        def _augmentation(image):
            width, height, channel = IMAGE_SHAPE
            x = tf.image.random_flip_left_right(image)
            x = tf.image.random_crop(
                x, tf.stack([int(width*0.8), int(height*0.8), channel]))
            x = tf.image.resize(x, (width, height))
            return x

        image1 = \
            tf.reshape(float_image1_raw, IMAGE_SHAPE, name='image1')
        image1 = _augmentation(image1)

        image2 = \
            tf.reshape(float_image2_raw, IMAGE_SHAPE, name='image2')
        image2 = _augmentation(image2)

        return ((image1, image2), label)
