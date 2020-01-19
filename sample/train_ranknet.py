from ImageRankNet import RankNet, dataset
import tensorflow as tf
import os
import sys
from mapper import SampleMapper
from config import tfrecords_dir, weights_dir, logs_dir, IMAGE_SHAPE

sys.path.append(os.getcwd())


if __name__ == "__main__":
    ranknet = RankNet(IMAGE_SHAPE)

    mapper = SampleMapper()
    train_dataset = dataset.make_dataset(
        str(tfrecords_dir/'train.tfrecord'), mapper, 1, 'train')
    valid_dataset = dataset.make_dataset(
        str(tfrecords_dir/'validation.tfrecord'), mapper, 1, 'validation')

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_dir /
                'weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5'),
            save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir),
                                       write_graph=True)
    ]

    ranknet.train(train_dataset, valid_dataset, callback_list=callback_list)
