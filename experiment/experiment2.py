import sys

sys.path.append('../data')
sys.path.append('../networks')
sys.path.append('../test')
sys.path.append('../utils')

from ModelBuilder import ModelBuilder
import tensorflow as tf
from dataset import DataSetV2


class Experiment2:
    """
    Experiment2 plain nn application on short-reach system, in this experiment the
    """
    def __init__(self, symbol_win_size=21):
        # sampling per symbol, typically used for initializing the dataset
        self.samples_per_symbol = 16

        # input win size (how many symbols should be put into the model)
        self.win_size = symbol_win_size

        # Categorical Cross Entropy (cce) function pointer
        self.cce = tf.keras.losses.CategoricalCrossentropy()

        # the model
        self.model = self.build_nn(self.win_size)

        # the dataset
        self.dataset = DataSetV2(self.win_size, batch_size=20)

        # the optimizer
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        # the training counter in batch sense
        self.counter = 0

    @staticmethod
    def build_nn(win_sz: int):
        """
        build the classifier of the
        :param win_sz:
        :return:
        """
        builder = ModelBuilder(win_sz)

        model = builder \
            .build('fc', 2, [50, 50]) \
            .build('fc', 1, [4], activation_list=['softmax']) \
            .to_model()

        return model

    @tf.function
    def train_one_step(self, tx, gt):
        """
        train the model one step.
        :param tx: tx wave signal, normalized to [-1, 1]
        :param gt: gt in one-hot coding
        :return: None
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:

            tape.watch(self.model.trainable_variables)

            pred = self.model(tx, training=True)
            cce_loss = self.cce(gt, pred)

            gradient = tape.gradient(cce_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

            if self.counter % 1000 == 0:
                print("[info]: counter: ", self.counter, " cce_loss: ", cce_loss)

    def start_train_task(self):
        """
        start the training task.
        :return: None
        """
        for _, rx, gt in self.dataset:
            self.counter += 1
            self.train_one_step(rx, gt)

            if self.counter % 5000 == 0:
                self.save_model()

    def save_model(self):
        self.model.save('../save/' + 'plain_nn_' + str(self.counter) + '.h5')


if __name__ == '__main__':
    e = Experiment2()
    e.start_train_task()
