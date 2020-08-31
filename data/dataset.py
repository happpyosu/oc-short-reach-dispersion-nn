import tensorflow as tf
import numpy as np
import random


class Dataset:
    def __init__(self, win_size: int, base_dir='../dataset/', train_times=1000, batch_size=20):
        # tf dataset
        self.dataset = tf.data.TextLineDataset(tf.data.Dataset.list_files(base_dir + '*.txt')).\
            map(lambda x: tf.numpy_function(func=Dataset.str2float, inp=[x], Tout=tf.float32))

        # window size
        self.win_size = win_size

        # window cache
        self.cache = list()

        # init_cache
        self.init_cache()

        # max valid cursor of the window start
        self.max_cursor = len(self.cache) - self.win_size

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
        print("[info]: start init dataset cache...")
        for e in self.dataset.as_numpy_iterator():
            self.cache.append(e)
        print("[info]: Done init cache, cache length: " + str(len(self.cache)))

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

    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.training_times:
            raise StopIteration
        ret = None
        for i in range(self.batch_size):
            cur = random.randint(0, self.max_cursor)
            cut = self.cache[cur:cur+self.win_size]
            tensor = tf.expand_dims(tf.convert_to_tensor(cut, dtype=tf.float32), axis=0)
            if ret is None:
                ret = tensor
            else:
                ret = tf.concat([ret, tensor], axis=0)

        return ret

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
            cut = self.cache[cur:cur+self.win_size]
            self.fixed_win = tf.expand_dims(tf.convert_to_tensor(cut, dtype=tf.float32), axis=0)

        return self.fixed_win


if __name__ == '__main__':
    dataset = Dataset(100, train_times=10)

    for a in dataset:
        print(a)
    # a = next(iter)
    # split1, split2 = tf.split(a, 2, axis=-1)
    # print(split1)
    # print(split2)
