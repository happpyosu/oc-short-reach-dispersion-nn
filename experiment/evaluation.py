import sys
sys.path.append('../data')

from evalmodel import ModelEvaluator, Metric
from experiment1 import Experiment1
from experiment2 import Experiment2
from experiment3 import Experiment3
from dataset import DataSetV1, DataSetV2


class Evaluation1:
    """
    The Evaluation of the Experiment1
    """
    def __init__(self, weight_filename):

        # winsize
        win_sz = 23

        # model to evaluate
        self.model = Experiment1.build_cleaner(win_size=16 * (win_sz - 1))
        self.model.load_weights('../save/' + weight_filename)

        # dataset
        self.dataset = DataSetV1(win_sz, sample_per_sym=16, test_mode=True)

        # model evaluator
        self.evaluator = ModelEvaluator(model=self.model, dataset=self.dataset)
        # self.evaluator.add_metric(Metric.AVG_MSE)
        self.evaluator.add_metric(Metric.BER)

    def eval(self):
        self.evaluator.do_eval()


class Evaluation3:
    """
    The evaluation of Experiment3
    """
    def __init__(self, weight_filename):
        # win-size
        win_sz = 11

        # model
        self.model = Experiment3.build_cleaner(win_size=16 * (win_sz - 1))

        # load weight
        self.model.load_weights('../save/' + weight_filename)

        # dataset
        self.dataset = DataSetV1(win_sz, sample_per_sym=16, test_mode=True)

        # model evaluator
        self.evaluator = ModelEvaluator(model=self.model, dataset=self.dataset)

        self.evaluator.add_metric(Metric.BER_REGRESSION)

    def eval(self):
        self.evaluator.do_eval()


class Evaluation2:
    """
    The Evaluation of the Experiment2
    """
    def __init__(self, weight_filename):
        # win size
        win_sz = 11

        # model to evaluate
        self.model = Experiment2.build_nn(win_sz)
        self.model.load_weights('../save/' + weight_filename)

        # dataset
        self.dataset = DataSetV2(win_sz, test_mode=True)

        # model evaluator
        self.evaluator = ModelEvaluator(model=self.model, dataset=self.dataset)
        self.evaluator.add_metric(Metric.BER_SOFTMAX)
        # self.evaluator.add_metric(Metric.AVG_MSE)

    def eval(self):
        self.evaluator.do_eval()


if __name__ == '__main__':
    Evaluation1('cleaner_final.h5').eval()
    # Evaluation2('cleaner_final.h5').eval()
    # Evaluation3('cleaner_final.h5').eval()