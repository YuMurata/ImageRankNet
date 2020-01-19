import tensorflow as tf
from pathlib import Path
from .mapper import Mapper
import typing

ShapeTuple = typing.Tuple[int, int, int]

SCOPE = 'ranknet_dataset'


class DatasetException(Exception):
    pass


def make_dataset(dataset_file_path: str,
                 mapper: Mapper,
                 batch_size: int,
                 name: str) -> tf.data.TFRecordDataset:

    dataset_file_path = Path(dataset_file_path)
    if not dataset_file_path.exists():
        raise DatasetException(f'{str(dataset_file_path)} is not found')

    with tf.name_scope(f'{name}_{SCOPE}'):
        dataset = \
            tf.data.TFRecordDataset(str(dataset_file_path)) \
            .map(mapper.map_example) \
            .shuffle(batch_size) \
            .batch(batch_size) \
            .repeat()

    return dataset
