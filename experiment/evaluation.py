import sys
sys.path.append('../data')

from evalmodel import ModelEvaluator, Metric
from experiment1 import Experiment1
from experiment2 import Experiment2
from dataset import TestDataSetV2

class Evaluation1:
    """
    The Evaluation of the Experiment1
    """
    def __init__(self, weight_path):
        # build cleaner evaluator
        self.cleaner_evaluator = ModelEvaluator(model=Experiment1.build_cleaner(11 * 32))
        self.cleaner_evaluator.load_weight(weight_path)
        self.cleaner_evaluator.add_metric(Metric.BER)

    def eval(self):
        self.cleaner_evaluator.do_eval()


class Evaluation2:
    """
    The Evaluation of the Experiment2
    """
    def __init__(self, weight, test_set):
        # build model
        self.model_evaluator = ModelEvaluator(model=Experiment2.build_nn(21))
        self.model_evaluator.set_dataset(TestDataSetV2(win_size=21, dataset_filename=test_set))
        self.model_evaluator.add_metric(Metric.BER_SOFTMAX)
        self.model_evaluator.load_weight(weight)

    def eval(self):
        self.model_evaluator.do_eval()


if __name__ == '__main__':
    Evaluation2('plain_nn_100000.h5', '28G_dataout10000.mat.txt').eval()