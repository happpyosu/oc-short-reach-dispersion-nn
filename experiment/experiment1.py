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
        self.beta = 0.001
        self.gamma = 1

        # training epoch
        self.epoch = 100

        # build the cleaner
        self.cleaner = Experiment1.build_cleaner(self.win_size)

        # build the polluter
        self.polluter = Experiment1.build_polluter(self.win_size)

        # build the cleaner critic
        self.cleaner_critic = self._build_critic()

        # build the polluter critic
        self.polluter_critic = self._build_critic()

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
        self.polluter_optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.cleaner_optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.polluter_critic_optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.cleaner_critic_optimizer = tf.keras.optimizers.Adam(lr_schedule)

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

    @staticmethod
    def build_polluter(win_size: int):
        builder = ModelBuilder(win_size)

        model = builder \
            .build('fc', 4, [25, 50, 100, 200]) \
            .build('fc', 4, [200, 100, 50, 25]) \
            .build('fc', 1, [win_size], activation_list=['none']) \
            .to_model()

        return model

    def _build_critic(self):
        critic = CriticFactory.get_conv_critic(self.win_size, 6, filter_list=[64, 128, 256, 128, 64, 32])
        return critic

    def start_train_task(self):

        self.print_train_context()
        print("\033[1;32m" + '[info]: (Experiment1) training...' + " \033[0m")
        for tx, rx, _ in self.dataset:
            self.counter += 1
            self.train_one_step(tx, rx)

            if self.counter % 10000 == 0:
                self.cleaner.save_weights(filepath='../save/cleaner_' + str(self.counter) + '.h5')
                self.polluter.save_weights(filepath='../save/polluter_' + str(self.counter) + '.h5')
                self.output_middle_result(counter=self.counter)

        self.cleaner.save_weights(filepath='../save/cleaner_' + 'final' + '.h5')
        self.polluter.save_weights(filepath='../save/polluter_' + 'final' + '.h5')
        self.output_middle_result(counter=self.counter)

    def print_train_context(self):
        with open('../save/train_context.txt', 'w') as file:
            file.writelines("samples per symbol: " + str(self.samples_per_symbol) + "\n")
            file.writelines("symbol window length: " + str(self.symbols_win) + "\n")
            file.writelines("loss weights: " + "(l2 distance loss) α = " + str(self.alpha)
                            + " (cyclic consistency loss) β = " + str(self.beta)
                            + " (gan loss) γ = " + str(self.gamma) + "\n")

    def output_middle_result(self, counter):
        fixed_win = self.dataset.get_fixed_win()
        tx = fixed_win[0]
        rx = fixed_win[1]
        clean_wave = self.cleaner(rx)
        dirty_wave = self.polluter(tx)

        tx = tf.squeeze(tx, axis=0)
        rx = tf.squeeze(rx, axis=0)
        clean_wave = tf.squeeze(clean_wave, axis=0)
        dirty_wave = tf.squeeze(dirty_wave, axis=0)

        pltUtils.plot_wave_tensors(tx, rx, clean_wave, dirty_wave, legend_list=['tx', 'rx', 'clean-wave', 'dirty-wave'],
                                   is_save_file=True, file_name=str(self.counter) + '.jpg')

    @tf.function
    def train_one_step(self, tx, rx):
        with tf.GradientTape(watch_accessed_variables=False) as d1_tape, \
                tf.GradientTape(watch_accessed_variables=False) as d2_tape:
            # do watch tf model
            # polluter_tape.watch(self.polluter.trainable_variables)
            # cleaner_tape.watch(self.cleaner.trainable_variables)
            d1_tape.watch(self.polluter_critic.trainable_variables)
            d2_tape.watch(self.cleaner_critic.trainable_variables)

            # -----------------------------------------Step 1: train two critics----------------------------------------
            fake_dirty_wave = self.polluter(tx, training=True)

            critic_on_fake = self.polluter_critic(fake_dirty_wave, training=True)
            critic_on_real = self.polluter_critic(rx, training=True)
            critic_loss = tf.reduce_mean(MSE(critic_on_fake, tf.zeros_like(critic_on_fake)) + \
                                         MSE(critic_on_real, tf.ones_like(critic_on_real)), keepdims=True)

            gradient_of_polluter_critic = d1_tape.gradient(critic_loss, self.polluter_critic.trainable_variables)

            self.polluter_critic_optimizer.apply_gradients(zip(gradient_of_polluter_critic,
                                                               self.polluter_critic.trainable_variables))

            fake_clean_wave = self.cleaner(rx, training=True)
            critic_on_fake2 = self.cleaner_critic(fake_clean_wave, training=True)
            critic_on_real2 = self.cleaner_critic(tx, training=True)
            critic_loss2 = tf.reduce_mean(MSE(critic_on_fake2, tf.zeros_like(critic_on_fake2)) + \
                                          MSE(critic_on_real2, tf.ones_like(critic_on_real2)), keepdims=True)

            gradient_of_cleaner_critic = d2_tape.gradient(critic_loss2, self.cleaner_critic.trainable_variables)
            self.cleaner_critic_optimizer.apply_gradients(zip(gradient_of_cleaner_critic,
                                                              self.cleaner_critic.trainable_variables))

        with tf.GradientTape() as polluter_tape, tf.GradientTape() as cleaner_tape:

            # -----------------------------------Step 2: polluter -> cleaner (cycle1)-----------------------------------
            # let polluter pollute the tx signal
            dirty_wave = self.polluter(tx, training=True)

            # the generated dirty_wave should be close to the real rx signal in l2-distance sense.
            polluter_l2_loss = tf.reduce_mean((rx - dirty_wave) ** 2)

            # score on fake "dirty wave"
            critic_on_fake_dirty_wave = self.polluter_critic(dirty_wave, training=True)

            # the polluter should "fool" the polluter critic
            polluter_critic_loss = tf.reduce_mean(MSE(critic_on_fake_dirty_wave,
                                                      tf.ones_like(critic_on_fake_dirty_wave)), keepdims=True)

            # let the cleaner clean the dirty wave
            after_clean = self.cleaner(dirty_wave, training=True)

            # the cyclic consistency loss
            polluter_cyclic_loss = tf.reduce_mean((tx - after_clean) ** 2, keepdims=True)

            # total loss
            total_polluter_loss = self.alpha * polluter_l2_loss + \
                                  self.beta * polluter_critic_loss + self.gamma * polluter_cyclic_loss

            # update gradient
            gradients_of_polluter = polluter_tape.gradient(total_polluter_loss, self.polluter.trainable_variables)
            gradients_of_cleaner = cleaner_tape.gradient(total_polluter_loss, self.cleaner.trainable_variables)
            self.polluter_optimizer.apply_gradients(zip(gradients_of_polluter, self.polluter.trainable_variables))
            self.cleaner_optimizer.apply_gradients(zip(gradients_of_cleaner, self.cleaner.trainable_variables))

        with tf.GradientTape() as polluter_tape, tf.GradientTape() as cleaner_tape:
            # -----------------------------------------Step 3: cleaner -> polluter (cycle2)-----------------------------
            # let the cleaner clean the rx signal
            clean_wave = self.cleaner(rx, training=True)

            # the generated clean wave should be close to the real tx signal in l2-distance sense.
            cleaner_l2_loss = tf.reduce_mean((tx - clean_wave) ** 2, keepdims=True)

            # score on fake "clean wave"
            critic_on_fake_clean_wave = self.cleaner_critic(clean_wave, training=True)

            # the cleaner should "fool" the cleaner critic
            cleaner_critic_loss = tf.reduce_mean(MSE(critic_on_fake_clean_wave,
                                                     tf.ones_like(critic_on_fake_clean_wave)), keepdims=True)

            # let the polluter pollute the clean wave
            after_pollute = self.polluter(clean_wave, training=True)

            # the cyclic consistency loss
            cleaner_cyclic_loss = tf.reduce_mean((rx - after_pollute) ** 2, keepdims=True)

            # total loss
            total_cleaner_loss = self.alpha * cleaner_l2_loss + \
                                 self.beta * cleaner_critic_loss + self.gamma * cleaner_cyclic_loss

            gradients_of_cleaner = cleaner_tape.gradient(total_cleaner_loss, self.cleaner.trainable_variables)
            gradients_of_polluter = polluter_tape.gradient(total_cleaner_loss, self.polluter.trainable_variables)
            self.cleaner_optimizer.apply_gradients(zip(gradients_of_cleaner, self.cleaner.trainable_variables))
            self.polluter_optimizer.apply_gradients(zip(gradients_of_polluter, self.polluter.trainable_variables))
            # -----------------------------------------Step 4: print some info------------------------------------------
            if self.counter % 20 == 0:
                print("[info]: counter: " + str(self.counter) +
                      " polluter_critic_loss: " + str(critic_loss) +
                      " cleaner_critic_loss: " + str(critic_loss2) +
                      " total_polluter_loss: " + str(total_polluter_loss) +
                      " total_cleaner_loss: " + str(total_cleaner_loss))

    def print_experiment_context(self):
        print("[info]: ===========experiment1 context===========")
        print("[info]: samples per symbol: " + str(self.samples_per_symbol))
        print("[info]: symbols windows: " + str(self.symbols_win))
        print("[info]: loss weights: " + "alpha: " + str(self.alpha) +
              "[info]: beta: " + str(self.beta) + " gamma: " + str(self.gamma))

        print("\n")
        print("cleaner summary: ")
        self.cleaner.summary()
        print("\n")

        print("\n")
        print("polluter summary: ")
        self.polluter.summary()
        print("\n")

        print("\n")
        print("cleaner_critic summary: ")
        self.cleaner_critic.summary()
        print("\n")

        print("\n")
        print("polluter_critic summary: ")
        self.polluter_critic.summary()
        print("\n")


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(gpu_index=2)
    exp = Experiment1(symbol_win_size=7)
    exp.print_experiment_context()
    exp.start_train_task()


