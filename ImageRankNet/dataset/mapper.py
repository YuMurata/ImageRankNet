from abc import ABCMeta, abstractmethod
import typing
import tensorflow as tf


class Mapper(metaclass=ABCMeta):
    '''
    TFRecordDataset用にtfrecordsをパースする

    implement
    ---
    map_example
        param:
            example_proto : tf.train.Example

        return:
            ((tf.Tensor, tf.Tensor) tf.int32)
            - same tf.Tensor size 

    '''

    @abstractmethod
    def map_example(self, example_proto: tf.train.Example) -> ((tf.Tensor, tf.Tensor), tf.int32):
        pass
