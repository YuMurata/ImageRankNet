from .dataset import ShapeTuple
import tensorflow as tf
from pathlib import Path
import numpy as np
from .grad_cam import GradCam
from .evaluate_network import build_evaluate_network
import typing
from PIL.Image import Image
from .structure import Structure

ImageList = typing.List[Image]


class RankNet:
    SCOPE = 'predict_model'
    TRAINABLE_MODEL_FILE_NAME = 'trainable_model.h5'

    class ImageInfo:
        def __init__(self, image_shape: ShapeTuple):
            self.shape = image_shape
            self.width, self.height, self.channel = image_shape
            self.size = (self.width, self.height)

    def _image_to_array(self, image: Image) -> np.array:
        resized_image = image.resize(self.image_info.size)
        return np.asarray(resized_image).astype(np.float32)/255

    def __init__(self, image_shape: ShapeTuple, *, use_vgg16: bool = False):
        self.image_info = RankNet.ImageInfo(image_shape)

        with tf.name_scope(RankNet.SCOPE):
            evaluate_network = build_evaluate_network(
                image_shape, use_vgg16=use_vgg16)
            self.grad_cam = GradCam(evaluate_network, self.image_info.size)

            left_input = tf.keras.Input(shape=image_shape)
            right_input = tf.keras.Input(shape=image_shape)

            left_output = evaluate_network(left_input)
            right_output = evaluate_network(right_input)

            concated_output = \
                tf.keras.layers.Concatenate()([left_output, right_output])

            with tf.name_scope('predictable_model'):
                self.predictable_model = tf.keras.Model(inputs=left_input,
                                                        outputs=left_output)
            with tf.name_scope('trainable_model'):
                self.trainable_model = tf.keras.Model(inputs=[left_input,
                                                              right_input],
                                                      outputs=concated_output)

            loss = \
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)
            self.trainable_model.compile(optimizer='adam', loss=loss)

            self.structure = Structure(
                self.predictable_model, self.trainable_model, evaluate_network)

    def train(self, dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset,
              *,
              callback_list: typing.List[typing.Callable] = [],
              epochs: int = 10,
              steps_per_epoch: int = 30):

        self.trainable_model.fit(dataset, epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callback_list,
                                 validation_data=valid_dataset,
                                 validation_steps=10)

    def save(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        self.trainable_model.save_weights(
            str(Path(save_dir_path) /
                RankNet.TRAINABLE_MODEL_FILE_NAME))

    def load(self, load_file_path: str):
        self.trainable_model.load_weights(load_file_path)

    def predict(self, data_list: ImageList) -> np.array:
        image_array_list = np.array([self._image_to_array(data)
                                     for data in data_list])

        return self.predictable_model.predict(image_array_list)
