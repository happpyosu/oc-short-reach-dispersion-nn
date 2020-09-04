import sys

sys.path.append('../utils')
import tensorflow as tf
import numpy as np
import random
from utils.plotutils import PlotUtils


class AbstractDataset:
    """
    AbstractDataset class for build the pipeline between the txt file and tf.Dataset
    """

    def __init__(self, win_size: int, base_dir='../dataset/'):
        # tf dataset
        self.dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files(base_dir + '*.txt')). \
            map(lambda x: tf.numpy_function(func=AbstractDataset.str2float, inp=[x], Tout=tf.float32))

        # window size
        self.win_size = win_size

    @staticmethod
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


class TrainingDataset(AbstractDataset):
    """
    Training Dataset class, for generating data during training a dataset
    """

    def __init__(self, win_size: int, base_dir='../dataset/', train_times=1000, batch_size=20):

        # call super class for init
        super().__init__(win_size, base_dir)

        # tx cache
        self.tx_cache = list()

        # rx cache
        self.rx_cache = list()

        # init_cache
        self.init_cache()

        # max valid cursor of the window start
        self.max_cursor = len(self.tx_cache) - self.win_size

        # training times
        self.training_times = train_times

        # counter
        self.counter = 0

        # batch size
        self.batch_size = batch_size

        # fixed window size batch_size
        self.fixed_win = None

    def init_cache(self):
        # init the cache
        print("[info]: start init training dataset cache...")
        iterator = self.dataset.as_numpy_iterator()
        # load tx cache
        self.tx_cache = next(iterator)
        self.rx_cache = next(iterator)

        if len(self.tx_cache) != len(self.rx_cache):
            raise ValueError("The tx length is not consistent with the rx length in training dataset")

        print("[info]: Done init training cache, cache length: " + str(len(self.tx_cache)))

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.training_times:
            raise StopIteration
        ret_tx = None
        ret_rx = None

        for i in range(self.batch_size):
            cur = random.randint(0, self.max_cursor)
            cut_tx = self.tx_cache[cur:cur + self.win_size]
            cut_rx = self.rx_cache[cur:cur + self.win_size]
            tensor_tx = tf.expand_dims(tf.convert_to_tensor(cut_tx, dtype=tf.float32), axis=0)
            tensor_rx = tf.expand_dims(tf.convert_to_tensor(cut_rx, dtype=tf.float32), axis=0)
            if ret_tx is None:
                ret_tx = tensor_tx
                ret_rx = tensor_rx
            else:
                ret_tx = tf.concat([ret_tx, tensor_tx], axis=0)
                ret_rx = tf.concat([ret_rx, tensor_rx], axis=0)
        return ret_tx, ret_rx

    def reset_iter_context(self):
        """
        reset the iteration context.
        :return: None
        """
        self.counter = 0

    def get_fixed_win(self):
        """
        get the fixed sample, the batch size of the fixed window sample is always 1
        :return: tf.Tensor
        """
        # lazy init the fixed_win
        if self.fixed_win is None:
            # cur = random.randint(0, self.max_cursor)
            cur = 8888
            cut_tx = self.tx_cache[cur:cur + self.win_size]
            cut_rx = self.rx_cache[cur:cur + self.win_size]
            fixed_win_tx = tf.expand_dims(tf.convert_to_tensor(cut_tx, dtype=tf.float32), axis=0)
            fixed_win_rx = tf.expand_dims(tf.convert_to_tensor(cut_rx, dtype=tf.float32), axis=0)

            self.fixed_win = (fixed_win_tx, fixed_win_rx)

        return self.fixed_win


class TestDataSet(AbstractDataset):
    """
    Test dataset for testing the trained tf Model
    """

    def __init__(self, win_size: int, base_dir='../testset'):
        super().__init__(win_size=win_size, base_dir=base_dir)

        self.cache = list()

    def init_cache(self):
        # init the cache
        print("[info]: start init training dataset cache...")
        for e in self.dataset.as_numpy_iterator():
            self.cache.append(e)
        print("[info]: Done init training cache, cache length: " + str(len(self.cache)))


if __name__ == '__main__':
    dataset = TrainingDataset(100, train_times=10)
    # a = next(iter)
    # split1, split2 = tf.split(a, 2, axis=-1)
    # print(split1)
    # print(split2)
