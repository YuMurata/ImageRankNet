import tensorflow as tf
import os
import sys
from mapper import SampleMapper
from config import tfrecords_dir, weights_dir, logs_dir, IMAGE_SHAPE
from PIL.Image import fromarray
import numpy as np

sys.path.append(os.getcwd())
from ImageRankNet import RankNet, dataset


if __name__ == "__main__":
    ranknet = RankNet(IMAGE_SHAPE)
    ranknet.load(r'sample\weights\weights.01-0.07-0.00.h5')
    
    white = np.ones(IMAGE_SHAPE, dtype=np.uint8)*255
    black = np.zeros(IMAGE_SHAPE, dtype=np.uint8) 

    predict = ranknet.predict(list(map(fromarray, [white, black])))

    print(predict)
