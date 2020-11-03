import sys
sys.path.append('../data')

from evalmodel import ModelEvaluator, Metric
from experiment1 import Experiment1
from experiment2 import Experiment2
from dataset import DataSetV1, DataSetV2


class Evaluation1:
    """
    The Evaluation of the Experiment1
    """
    def __init__(self, weight_filename):
        # model to evaluate
        self.model = Experiment1.build_cleaner(win_size=16 * 18)
        self.model.load_weights('../save/' + weight_filename)

        # dataset
        self.dataset = DataSetV1(19, sample_per_sym=16, test_mode=True)

        # model evaluator
        self.evaluator = ModelEvaluator(model=self.model, dataset=self.dataset)
        # self.evaluator.add_metric(Metric.AVG_MSE)
        self.evaluator.add_metric(Metric.BER)

    def eval(self):
        self.evaluator.do_eval()


class Evaluation2:
    """
    The Evaluation of the Experiment2
    """
    def __init__(self, weight_filename):
        # model to evaluate
        self.model = Experiment2.build_nn(21)
        self.model.load_weights('../save/' + weight_filename)

        # dataset
        self.dataset = DataSetV2(21, test_mode=True)

        # model evaluator
        self.evaluator = ModelEvaluator(model=self.model, dataset=self.dataset)
        # self.evaluator.add_metric(Metric.BER_SOFTMAX)
        self.evaluator.add_metric(Metric.AVG_MSE)

    def eval(self):
        self.evaluator.do_eval()


if __name__ == '__main__':
    Evaluation1('cleaner_160000.h5').eval()
    # Evaluation2('plain_nn_160000.h5').eval()