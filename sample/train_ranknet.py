import tensorflow as tf
import os
import sys
from mapper import SampleMapper
from config import tfrecords_dir, weights_dir, logs_dir, IMAGE_SHAPE
from evaluate_body import MyCNN

sys.path.append(os.getcwd())
from ImageRankNet import RankNet, dataset


if __name__ == "__main__":
    ranknet = RankNet(MyCNN())

    mapper = SampleMapper()
    train_dataset = dataset.make_dataset(
        str(tfrecords_dir / 'train.tfrecord'), mapper, 1, 'train')
    valid_dataset = dataset.make_dataset(
        str(tfrecords_dir / 'validation.tfrecord'), mapper, 1, 'validation')

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_dir /
                'sample.h5'),
            save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir),
                                       write_graph=True)
    ]

    ranknet.train(train_dataset, valid_dataset, callback_list=callback_list)
