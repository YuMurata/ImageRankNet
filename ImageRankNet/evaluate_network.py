import tensorflow as tf
from .dataset import ShapeTuple
from abc import ABCMeta, abstractmethod


class EvaluateBody(metaclass=ABCMeta):
    '''
    implement
    ---
    def build(self) -> tf.keras.Model

    build your cnn
    '''

    def __init__(self, image_shape: tuple):
        self.image_shape = image_shape

    @abstractmethod
    def build(self) -> tf.keras.Model:
        pass


def _build_my_cnn(input_shape: ShapeTuple):
    layers = tf.keras.layers

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=5,
                            padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(filters=64, kernel_size=5,
                            padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    return model


def _build_vgg16(input_shape: ShapeTuple):
    vgg16 = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in vgg16.layers[:15]:
        layer.trainable = False

    return vgg16


def build_evaluate_network(evaluate_body: EvaluateBody) -> tf.keras.Model:
    convolution_layer = evaluate_body.build()

    top_layer = tf.keras.Sequential()
    top_layer.add(tf.keras.layers.Flatten(
        input_shape=convolution_layer.output_shape[1:]))
    top_layer.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    top_layer.add(tf.keras.layers.Dropout(rate=0.5))
    top_layer.add(tf.keras.layers.Dense(units=1))

    model = tf.keras.Model(inputs=convolution_layer.input,
                           outputs=top_layer(convolution_layer.output))
    return model
