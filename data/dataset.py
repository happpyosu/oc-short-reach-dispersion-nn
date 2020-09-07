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

    def __init__(self, win_size: int, base_dir='../dataset/', batch_size: int = 20):
        # tf dataset
        self.dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files(base_dir + '*.txt')). \
            map(lambda x: tf.numpy_function(func=AbstractDataset.str2float, inp=[x], Tout=tf.float32))

        # window size
        self.win_size = win_size

        # batch size
        self.batch_size = batch_size

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

    def get_win_size(self):
        """
        return the win size of the dataset
        :return: win size of the dataset
        """
        return self.win_size


class TrainingDataset(AbstractDataset):
    """
    Training Dataset class, for generating data during training a dataset
    """

    def __init__(self, win_size: int, base_dir='../dataset/', train_times=1000, batch_size=20):

        # call super class for init
        super().__init__(win_size, base_dir, batch_size)

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

    def __init__(self, win_size: int, base_dir='../testset/', batch_size: int = 1):
        super().__init__(win_size=win_size, base_dir=base_dir, batch_size=batch_size)

        # tx signal cache
        self.tx_cache = list()

        # rx signal cache
        self.rx_cache = list()

        # symbol center position list (used for fast locating position evaluation window)
        self.pos_list = list()

        # ground truth symbol (ranged in [-3, -1, 1, 3])
        self.gt = list()

        # call init cache to init tx_cache, rx_cache, pos_list, gt
        self._init_cache()

        # max cursor in pos list
        self.win_range = self._init_win_range()

        # the cursor, recording the current position in pos_list, init by lo in win_range
        self.cursor = self.win_range[0]

    def _init_win_range(self):
        """
         init the range of the evaluation window
        :return:
        """
        lb = - (self.win_size / 2 - 1)
        up = self.win_size / 2

        lo = 0
        hi = len(self.pos_list) - 1

        while lo < len(self.pos_list) and self.pos_list[lo] + lb < 0:
            lo += 1

        while hi >= 0 and self.pos_list[hi] + up >= len(self.tx_cache):
            hi -= 1

        if lo >= hi:
            raise ValueError("invalid status of win range in test set, got lo = " + str(lo) +
                             ", and hi = " + str(hi) + " please check the pos_list in the test set")

        return lo, hi

    def _init_cache(self):
        """
        init caches for fast generating tf.tensor
        :return:
        """
        # init the cache
        print("[info]: start init testing dataset cache, sampling pos list and gt list...")
        iterator = self.dataset.as_numpy_iterator()
        self.tx_cache = next(iterator)
        self.rx_cache = next(iterator)
        self.pos_list = next(iterator)
        self.gt = next(iterator)

        if len(self.tx_cache) != len(self.rx_cache):
            raise ValueError("The tx length is not consistent with the rx length in test dataset")

        print("[info]: Done init testing cache, cache length: " + str(len(self.tx_cache)))

    def __iter__(self):
        self.cursor = self.win_range[0]
        return self

    def __next__(self):
        """
        iter method for generating the test data in a iterative manner
        :return: tx_tensor: tx_concat of the shape (batch_size, win_size), the batch_dim has been expanded
                 rx_tensor: rx_concat of the shape (batch_size, win_size), the batch_dim has been expanded
                 gt_concat: list of int type (length: batch_size), indicating the label
        """
        if self.cursor > self.win_range[1]:
            raise StopIteration

        batch_counter = 0
        tx_concat = None
        rx_concat = None
        gt_concat = list()
        while batch_counter < self.batch_size:

            if self.cursor > self.win_range[1]:
                break

            center = self.pos_list[self.cursor]
            lb = int(center - (self.win_size / 2 - 1))
            ub = int(center + self.win_size / 2 + 1)
            tx_tensor = tf.expand_dims(tf.convert_to_tensor(self.tx_cache[lb:ub], dtype=tf.float32), axis=0)
            rx_tensor = tf.expand_dims(tf.convert_to_tensor(self.rx_cache[lb:ub], dtype=tf.float32), axis=0)
            ground_truth = self.gt[center]

            if tx_concat is None and rx_concat is None:
                tx_concat = tx_tensor
                rx_concat = rx_tensor
                gt_concat.append(ground_truth)
            else:
                tx_concat = tf.concat([tx_concat, tx_tensor], axis=0)
                rx_concat = tf.concat([rx_concat, rx_tensor], axis=0)
                gt_concat.append(ground_truth)

            batch_counter += 1
            self.cursor += 1

        return tx_concat, rx_concat, gt_concat



if __name__ == '__main__':
    dataset = TrainingDataset(100, train_times=10)
    # a = next(iter)
    # split1, split2 = tf.split(a, 2, axis=-1)
    # print(split1)
    # print(split2)
