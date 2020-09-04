import sys
sys.path.append('../utils')
sys.path.append('../data')
from enum import Enum
import tensorflow as tf
from utils import plotutils
from data.dataset import TestDataSet


class ModelEvaluator:
    """
    Model Evaluator class is used to eval a trained nn model using several test set on different metrics.
    """
    def __init__(self, model: tf.keras.Model, win_size=31):
        # model to evaluate
        self.model = model

        # test dataset, test one b
        self.dataset = TestDataSet(win_size=win_size, base_dir='../testset')


class Metric(Enum):
    """
    enum metrics for the ModelEvaluator
    """
    # bit error rate, noting that the predicted signal is after decision
    BER = 0

    # average mse in the prediction, noting that the predicted signal is before decision
    AVG_MSE = 1

    # max set in the prediction, noting that the predicted signal is before decision
    MAX_MSE = 2


class MetricProcessor:
    """
    base class of metric processor for evaluating different metrics
    """
    def __init__(self, metric_type: int, model: tf.keras.Model, inp: list, oup: list):
        self.metric_type = metric_type
        self.model = model
        self.inp = inp
        self.oup = oup

    def process(self):
        """
        template method that should be override by its child class
        :return:
        """
        pass


class BERMetricProcessor(MetricProcessor):
    """
    ber metric processor
    """
    def __init__(self, metric_type: int, model: tf.keras.Model, inp: list, oup: list):
        super().__init__(metric_type, model, inp, oup)

    def process(self):
        """
        do process the inp and calculate the ber metric
        :return: None
        """




