import sys

sys.path.append('../data')
sys.path.append('../networks')
sys.path.append('../test')
sys.path.append('../utils')

from ModelBuilder import ModelBuilder
from Networks import CriticFactory
import tensorflow as tf
from dataset import DataSetV1
from tensorflow.keras.losses import MSE
from plotutils import PlotUtils as pltUtils
import gpuutils


class Experiment1:
    """
        Experiment one: training a bidirectional translation model for short-reach system
    """
    def __init__(self, symbol_win_size=7):
        # experiment context info
        self.samples_per_symbol = 16
        self.symbols_win = symbol_win_size
        self.win_size = self.samples_per_symbol * (self.symbols_win - 1)

        # alpha for l2 loss, beta for gan loss, gamma for cyclic-consistency loss
        self.alpha = 1

        # training epoch
        self.epoch = 100

        # build the cleaner
        self.cleaner = Experiment1.build_cleaner(self.win_size)

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

        # optimizers
        self.cleaner_optimizer = tf.keras.optimizers.Adam(lr_schedule)

        # counter
        self.counter = 0

    @staticmethod
    def build_cleaner(win_size: int):
        builder = ModelBuilder(win_size)

        model = builder \
            .build('fc', 4, [25, 50, 100, 200]) \
            .build('fc', 4, [200, 100, 50, 25]) \
            .build('fc', 1, [win_size], activation_list=['none']) \
            .to_model()

        return model

    def start_train_task(self):

        self.print_train_context()
        print("\033[1;32m" + '[info]: (Experiment1) training...' + " \033[0m")
        for tx, rx, _ in self.dataset:
            self.counter += 1
            self.train_one_step(tx, rx)

            if self.counter % 10000 == 0:
                self.cleaner.save_weights(filepath='../save/cleaner_' + str(self.counter) + '.h5')

        self.cleaner.save_weights(filepath='../save/cleaner_' + 'final' + '.h5')

    def print_train_context(self):
        with open('../save/train_context.txt', 'w') as file:
            file.writelines("samples per symbol: " + str(self.samples_per_symbol) + "\n")
            file.writelines("symbol window length: " + str(self.symbols_win) + "\n")

    @tf.function
    def train_one_step(self, tx, rx):
        with tf.GradientTape() as cleaner_tape:
            # -----------------------------------------Step 3: cleaner -> polluter (cycle2)-----------------------------
            # let the cleaner clean the rx signal
            clean_wave = self.cleaner(rx, training=True)

            # the generated clean wave should be close to the real tx signal in l2-distance sense.
            cleaner_l2_loss = tf.reduce_mean((tx - clean_wave) ** 2, keepdims=True)

            gradients_of_cleaner = cleaner_tape.gradient(cleaner_l2_loss, self.cleaner.trainable_variables)
            self.cleaner_optimizer.apply_gradients(zip(gradients_of_cleaner, self.cleaner.trainable_variables))

    def print_experiment_context(self):
        print("[info]: ===========experiment1 context===========")
        print("[info]: samples per symbol: " + str(self.samples_per_symbol))
        print("[info]: symbols windows: " + str(self.symbols_win))
        print("\n")
        print("cleaner summary: ")
        self.cleaner.summary()
        print("\n")


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(gpu_index=2)
    exp = Experiment1(symbol_win_size=7)
    exp.print_experiment_context()
    exp.start_train_task()