import sys

sys.path.append('../utils')

import tensorflow as tf
import numpy as np
import random


class AbstractDataset:
    """
    AbstractDataset class for build the pipeline between the txt file and tf.Dataset
    """

    def __init__(self, batch_size: int = 20, dataset_filename='*.txt', test_mode=False):

        self.test_mode = test_mode

        # dataset base dir
        if not test_mode:
            BASE_DIR = '../dataset/'
        else:
            BASE_DIR = '../testset/'

        # tf dataset
        self.dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files(BASE_DIR + dataset_filename)). \
            map(lambda x: tf.numpy_function(func=AbstractDataset.str2float, inp=[x], Tout=tf.float32))

        # batch size
        self.batch_size = batch_size

        # if the test_mode the batch size will always be set to one
        if test_mode:
            self.batch_size = 1

        # sample per symbol
        self.sample_per_symbol = 16

        # tx cache
        self.tx_cache = list()

        # rx cache
        self.rx_cache = list()

        # pos list
        self.pos_list = list()

        # gt list
        self.gt = list()

        # call init cache
        self._init_cache()

        # for the small-sized dataset (only 4096 symbols)
        self.tx_cache = self.tx_cache[0:4096 * self.sample_per_symbol]
        self.rx_cache = self.rx_cache[0:4096 * self.sample_per_symbol]
        self.pos_list = self.pos_list[0:4096]
        self.gt = self.gt[0:4096]

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

    def _init_cache(self):
        # init the cache
        print("\033[1;32m" + "[info]: (AbstractDataset) start init abstract dataset cache..." + " \033[0m")
        iterator = self.dataset.as_numpy_iterator()
        # load tx cache
        self.tx_cache = next(iterator)
        self.rx_cache = next(iterator)
        self.pos_list = next(iterator)
        self.gt = next(iterator)

        if len(self.tx_cache) != len(self.rx_cache):
            raise ValueError("The tx length is not consistent with the rx length in training dataset")

        print("\033[1;32m" + "[info]: (AbstractDataset) Done init cache, "
                             "cache length: " + str(len(self.tx_cache)))

    def get_batch_size(self):
        """
        return the batch size of the dataset
        :return: batch size of the dataset
        """
        return self.batch_size

    def get_sample_per_symbol(self):
        """
        get sample per symbol of the dataset
        :return: sample per symbol
        """
        return self.sample_per_symbol


class DataSetV1(AbstractDataset):

    """
    DataSetV1 is used to generate batched training or testing sample from window-to-window waveform
    """

    def __init__(self, sym_win_size: int, sample_per_sym: int = 16, batch_size: int = 20, dataset_filename='*.txt',
                 train_epoch=100, test_mode=False, eval_len=60000):
        """
        :param win_sym_size: symbol size for feeding into the model
        :param batch_size: batch size.
        :param dataset_filename: dataset file name
        :param train_epoch: epoch for training
        """

        # call super class to init
        super().__init__(batch_size, dataset_filename, test_mode=test_mode)

        # symbol window size
        self.sym_win_size = sym_win_size

        # sample per symbol
        self.sample_per_symbol = sample_per_sym

        # half span
        self.half_span = self.sym_win_size // 2

        # epoch
        self.epoch = train_epoch

        # counter
        self.counter = 0

        # max cursor in pos list, the win range is stored as a tuple [low, high]
        self.win_range = self._init_win_range()

        # training times counter
        self.training_times = ((self.win_range[1] - self.win_range[0] + 1) * self.epoch) // self.batch_size

        # step per epoch used for init the decay scheduler
        self.step_per_epoch = (self.win_range[1] - self.win_range[0] + 1) // self.batch_size

        # fixed win
        self.fixed_win = None

        # if the dataset is in test mode, the training_times will be set to the len of win_range
        if self.test_mode:
            print("\033[1;34m" + "[info]: (DataSetV1) Test Mode is On..." + "\033[0m")
            print("\033[1;34m" + "[info]: (DataSetV1) total evaluation step " + str(eval_len) + "\033[0m")
            self.training_times = eval_len
        else:
            print("\033[1;34m" + "[info]: (DataSetV1) Training Mode is On..." + "\033[0m")
            print("\033[1;32m" + "[info]: (DataSetV1) epochs: " + str(self.epoch) +
                  " ,total steps: " + str(self.training_times) + " \033[0m")

    def _init_win_range(self):
        """
         init the range of the window
        :return: lo: the lowest valid index in pos_list ensuring the model input window is full
                 hi: the highest valid index in pos_list ensuring the model input window is full
        """
        half = self.sym_win_size // 2
        lo = half
        hi = len(self.pos_list) - 1 - half

        if lo > hi:
            raise ValueError("[Error]: (DataSetV1) invalid pos list length")

        print("\033[1;32m" + "[info]: (DataSetV1) win range of DatasetV2: ", str((lo, hi)),
              ", total data windows length for one epoch: " +
              str(hi - lo + 1) + " \033[0m")

        return lo, hi

    def get_fixed_win(self):
        """
        get the fixed sample, the batch size of the fixed window sample is always 1
        :return: tf.Tensor
        """
        # lazy init the fixed_win
        if self.fixed_win is None:

            # '888' means a lot of money
            cur = 888
            lb = int(self.pos_list[cur - self.half_span] - 1)
            ub = int(self.pos_list[cur + self.half_span] - 1)

            cut_tx = self.tx_cache[lb:ub]
            cut_rx = self.rx_cache[lb:ub]
            fixed_win_tx = tf.expand_dims(tf.convert_to_tensor(cut_tx, dtype=tf.float32), axis=0)
            fixed_win_rx = tf.expand_dims(tf.convert_to_tensor(cut_rx, dtype=tf.float32), axis=0)

            self.fixed_win = (fixed_win_tx, fixed_win_rx)

        return self.fixed_win

    def __iter__(self):
        # reset the counter
        self.counter = 0
        return self

    def __next__(self):
        """
        iter method for generating the test data in a iterative manner
        :return: tx_tensor: tx_concat of the shape (batch_size, win_size), the batch_dim has been expanded
                 rx_tensor: rx_concat of the shape (batch_size, win_size), the batch_dim has been expanded
                 gt_concat: list of int type (length: batch_size), indicating the label
        """
        self.counter += 1

        if self.counter > self.training_times:
            raise StopIteration

        tx_concat = None
        rx_concat = None
        gt_concat = None

        for i in range(self.batch_size):
            cursor = random.randint(self.win_range[0], self.win_range[1])
            lb = int(self.pos_list[cursor - self.half_span] - 1)
            ub = int(self.pos_list[cursor + self.half_span] - 1)

            tx_tensor = tf.expand_dims(tf.convert_to_tensor(self.tx_cache[lb:ub], dtype=tf.float32), axis=0)
            rx_tensor = tf.expand_dims(tf.convert_to_tensor(self.rx_cache[lb:ub], dtype=tf.float32), axis=0)
            gt_tensor = tf.expand_dims(tf.convert_to_tensor(self.gt[cursor], dtype=tf.float32), axis=0)

            if tx_concat is None:
                tx_concat = tx_tensor
                rx_concat = rx_tensor
                gt_concat = gt_tensor
            else:
                tx_concat = tf.concat([tx_concat, tx_tensor], axis=0)
                rx_concat = tf.concat([rx_concat, rx_tensor], axis=0)
                gt_concat = tf.concat([gt_concat, gt_tensor], axis=0)

        return tx_concat, rx_concat, gt_concat

    def get_step_per_epoch(self):
        return self.step_per_epoch


class DataSetV2(AbstractDataset):
    """
    DataSetV2 only will return down-sampled data point in the __next__() function, which is distinct from the
    DataSetV1 that producing the continous wave-form.
    """
    def __init__(self, sym_win_size: int, sample_per_sym: int = 16, batch_size: int = 20, dataset_filename='*.txt',
                 train_epoch=100, test_mode=False):
        # tx signal cache
        super().__init__(batch_size=batch_size, dataset_filename=dataset_filename, test_mode=test_mode)

        # symbol window size
        self.sym_win_size = sym_win_size

        # half span
        self.half_span = int(self.sym_win_size // 2)

        # sample per symbol
        self.sample_per_sym = sample_per_sym

        # win_range in pos list
        self.win_range = self._init_win_range()

        # counter
        self.counter = 0

        # training epoch, we cal the average epoch under the given training times
        self.epoch = train_epoch

        # training times
        self.training_times = ((self.win_range[1] - self.win_range[0] + 1) * self.epoch) // self.batch_size

        # step per epoch used for init the decay scheduler
        self.step_per_epoch = (self.win_range[1] - self.win_range[0] + 1) // self.batch_size

        if self.test_mode:
            print("\033[1;34m" + "[info]: (DataSetV2) Test Mode is On..." + "\033[0m")
            self.training_times = (self.win_range[1] - self.win_range[0] + 1)

        print("\033[1;32m" + "[info]: (DataSetV2) epoch: " + str(self.epoch) +
              " ,total steps: " + str(self.training_times) + " \033[0m")

    def _init_win_range(self):
        """
        cal the win range of the dataset
        :return: lo and hi limit in the pos list.
        """
        half_span = self.half_span

        lo = half_span
        hi = len(self.pos_list) - 1 - half_span

        return lo, hi

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.training_times:
            raise StopIteration

        tx_concat = None
        rx_concat = None
        gt_concat = None

        for i in range(self.batch_size):
            cursor = random.randint(self.win_range[0], self.win_range[1])
            left = cursor - self.half_span
            right = cursor + self.half_span + 1
            pos = self.pos_list[left:right]

            cut_tx = [self.tx_cache[int(x) - 1] for x in pos]
            cut_rx = [self.rx_cache[int(x) - 1] for x in pos]
            cut_gt = [int((self.gt[cursor] + 3) // 2)]

            cut_gt = tf.one_hot(cut_gt, 4)
            cut_tx = tf.expand_dims(tf.convert_to_tensor(cut_tx, dtype=tf.float32), axis=0)
            cut_rx = tf.expand_dims(tf.convert_to_tensor(cut_rx, dtype=tf.float32), axis=0)

            if tx_concat is None:
                tx_concat = cut_tx
                rx_concat = cut_rx
                gt_concat = cut_gt
            else:
                tx_concat = tf.concat([tx_concat, cut_tx], axis=0)
                rx_concat = tf.concat([rx_concat, cut_rx], axis=0)
                gt_concat = tf.concat([gt_concat, cut_gt], axis=0)

        return tx_concat, rx_concat, gt_concat


if __name__ == '__main__':
    v1 = DataSetV1(7, batch_size=1)
    for (tx, rx, gt) in v1:
        tx = tf.squeeze(tx, axis=0)
        print(gt.numpy()[0])





