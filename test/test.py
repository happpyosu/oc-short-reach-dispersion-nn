import sys

sys.path.append('../utils')
import tensorflow as tf
import numpy as np
import random
from utils.plotutils import PlotUtils


def str2float(x: np.ndarray):
    """
    map function for mapping plain string data to np.array float32, which can be further converted to tf.Tensor
    :param x: input np.ndarray
    :return: np.ndarray
    """

    s = str(x)[2:-1]
    split = s.split(' ')
    data = np.array(list(map(eval, split)), dtype='float32')
    return data


dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files('../testset/' + '*.txt')). \
            map(lambda x: tf.numpy_function(func=str2float, inp=[x], Tout=tf.float32))

for e in dataset.as_numpy_iterator():
    print(e, len(e))


