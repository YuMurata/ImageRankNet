import os
import sys
import tensorflow as tf

sys.path.append(os.getcwd())
from ImageRankNet import EvaluateBody
from config import IMAGE_SHAPE


class MyCNN(EvaluateBody):
    def __init__(self):
        super().__init__(IMAGE_SHAPE)

    def build(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                         padding='same', activation='relu',
                                         input_shape=self.image_shape))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                         padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        return model
