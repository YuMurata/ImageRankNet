import tensorflow as tf
from pathlib import Path


class Structure:
    def __init__(self, predictable_model: tf.keras.Model, trainable_model: tf.keras.Model, evaluate_network: tf.keras.Model):
        self.predictable_model = predictable_model
        self.trainable_model = trainable_model
        self.evaluate_network = evaluate_network

    def save(self, save_dir_path: str):
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        tf.keras.utils.plot_model(self.predictable_model,
                                  str(save_dir_path/'predictable_model.png'),
                                  show_shapes=True)

        tf.keras.utils.plot_model(self.trainable_model,
                                  str(save_dir_path/'trainable_model.png'),
                                  show_shapes=True)

        tf.keras.utils.plot_model(self.evaluate_network,
                                  str(save_dir_path/'evaluate_network.png'),
                                  show_shapes=True)
