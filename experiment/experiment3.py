import sys
import tensorflow as tf

sys.path.append('../data')
sys.path.append('../networks')
sys.path.append('../test')
sys.path.append('../utils')

from dataset import DataSetV1
from ModelBuilder import ModelBuilder
import gpuutils


class Experiment3:
    """
    The experiment3: using only on output node to perform the signal regression.
    """

    def __init__(self, symbol_win_size=7):
        self.samples_per_symbol = 16
        self.symbols_win = symbol_win_size
        self.win_size = self.samples_per_symbol * (self.symbols_win - 1)

        # training epoch
        self.epoch = 100

        # the cleaner model
        self.cleaner = Experiment3.build_cleaner(self.win_size)

        # tf dataset
        self.dataset = DataSetV1(self.symbols_win, sample_per_sym=self.samples_per_symbol,
                                 train_epoch=self.epoch, batch_size=40)

        # learning rate decay policy
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=self.dataset.get_step_per_epoch() * self.epoch,
            decay_rate=1,
            staircase=False
        )

        # optimizer
        self.cleaner_optimizer = tf.keras.optimizers.Adam(lr_schedule)

        # counter
        self.counter = 0

        self.sample_index = int(((self.dataset.sym_win_size - 1) * self.dataset.sample_per_symbol) // 2)

    @staticmethod
    def build_cleaner(win_size: int):
        builder = ModelBuilder(win_size)

        model = builder \
            .build('fc', 4, [25, 50, 100, 200]) \
            .build('fc', 4, [200, 100, 50, 25]) \
            .build('fc', 1, [1], activation_list=['tanh']) \
            .to_model()

        return model

    @tf.function
    def train_one_step(self, tx, rx):
        """
        :param tx: tx signal with only one dimensions
        :param rx: rx signal with win-size dimensions
        :return: None
        """
        with tf.GradientTape() as tape:
            pred_tx = self.cleaner(rx, training=True)
            l2_loss = tf.reduce_mean((pred_tx - tx) ** 2)
            if self.counter % 500 == 0:
                print('l2-loss', l2_loss)
            grad = tape.gradient(l2_loss, self.cleaner.trainable_variables)
            self.cleaner_optimizer.apply_gradients(zip(grad, self.cleaner.trainable_variables))

    def start_train_task(self):
        print("\033[1;32m" + '[info]: (Experiment3) training...' + " \033[0m")
        for tx, rx, _ in self.dataset:
            self.counter += 1
            tx = tx[:, self.sample_index]
            tx = tf.expand_dims(tx, axis=1)
            self.train_one_step(tx, rx)

            if self.counter % 5000 == 0:
                self.cleaner.save_weights(filepath='../save/cleaner_' + str(self.counter) + '.h5')
        self.cleaner.save_weights(filepath='../save/cleaner_' + 'final' + '.h5')


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(gpu_index=0)
    exp = Experiment3(symbol_win_size=11)
    exp.start_train_task()
