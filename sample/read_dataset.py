import tensorflow as tf
from mapper import SampleMapper
from pathlib import Path
import os
import sys
from config import tfrecords_dir

sys.path.append(os.getcwd())
from ImageRankNet import dataset

if __name__ == "__main__":
    mapper = SampleMapper()

    train_dataset = dataset.make_dataset(
        str(tfrecords_dir/'train.tfrecord'), mapper, 1, 'train')

    for data in train_dataset.take(1):
        print(data)
