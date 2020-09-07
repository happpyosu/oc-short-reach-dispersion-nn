import sys
sys.path.append('../utils')
sys.path.append('../data')

from enum import Enum
import tensorflow as tf
from dataset import TestDataSet


class Metric(Enum):
    """
    enum metrics for the ModelEvaluator
    """
    # bit error rate, noting that the predicted signal is after decision
    BER = 0

    # average mse in the prediction, noting that the predicted signal is before decision
    AVG_MSE = 1


class ModelEvaluator:
    """
    Model Evaluator class is used to eval a trained nn model using several test set on different metrics.
    """
    def __init__(self, model: tf.keras.Model, symbol_win_size=11):
        # model to evaluate
        self.model = model

        # some meta data for building evaluation context
        self.samples_per_symbol = 32
        self.symbols_win = symbol_win_size
        self.win_size = self.samples_per_symbol * self.symbols_win

        # test dataset, test one b
        self.testset = TestDataSet(win_size=self.win_size, base_dir='../testset/')

        # metric processor list
        self.metric_processor_list = list()

    def add_metric(self, metric: Metric):
        """
        add a evaluation metric to this model evaluator, this function can be called in a chained manner.
        :param metric: Metric Enum object, see the definition.
        :return: None
        """
        if metric == Metric.BER:
            self.metric_processor_list.append(BERMetricProcessor(Metric.BER.value, self.model, self.testset))
        elif metric == Metric.AVG_MSE:
            self.metric_processor_list.append(AVGMSEMetricProcessor(Metric.AVG_MSE.value, self.model, self.testset))

        return self

    def do_eval(self):
        if len(self.metric_processor_list) == 0:
            print("[Warning]: the {do_eval} function is called, "
                  "but no metric processor is added to the model evaluator")
            return
        else:
            for processor in self.metric_processor_list:
                # directly call the processor's process method, since the evaluation context has been built before.
                processor.process()


class MetricProcessor:
    """
    base class of metric processor for evaluating different metrics
    """
    def __init__(self, metric_type: int, model: tf.keras.Model, test_set: TestDataSet):
        self.metric_type = metric_type
        self.model = model
        self.test_set = test_set

    def process(self):
        """
        template method that should be override by its child class
        :return: None
        """
        pass


class BERMetricProcessor(MetricProcessor):
    """
    ber metric processor
    """
    def __init__(self, metric_type: int, model: tf.keras.Model, test_set: TestDataSet):
        # directly call the super class to init the test context.
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the test dataset and calculate the ber metric
        :return: the BER performance of the model
        """
        right = 0
        error = 0
        for _, rx, gt in self.test_set:
            batch_sz = rx[0]
            pred = self.model(rx)
            res = self._decode_pam4(pred)
            res_map = [gt[i] == res[i] for i in range(len(gt))]
            right_num = sum(res_map)
            error_num = len(res_map) - right_num

            right += right_num
            error += error_num

        print("[info]: <BERMetricProcessor> total symbol: " + str(right + error) + " , right decision: " +
              str(right) + " ,error decision: " + str(error) + ", ber: " + str(right / (right + error)))

    def _decode_pam4(self, pred_tx, sig_range: tuple = (-1, 1)):
        """
        function used to decode the signal in sig_range (default (-1, 1)) to standard pam4 signal (-3, -1, 1, 3)
        :param pred_tx: tf.Tensor, basically the predicted result of the cleaner
        :return: a list of the decoding results, the length is same with the batch size of pred_tx
        """
        batch_sz = pred_tx.shape[0]
        sample_index = self.test_set.get_win_size() / 2 - 1
        res = list()
        for i in range(batch_sz):
            ds = float(pred_tx[i, int(sample_index)].numpy())

            if ds > 2. / 3:
                de = 3
            elif 2. / 3 >= ds > 0:
                de = 1
            elif 0 >= ds > -2. / 3:
                de = -1
            else:
                de = -3

            res.append(de)

        return res


class AVGMSEMetricProcessor(MetricProcessor):
    """
    AVG_MSE metric processor
    """
    def __init__(self, metric_type: int, model: tf.keras.Model, test_set: TestDataSet):
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the average accuracy on the test dataset
        :return:
        """
        avg_mse = []
        for tx, rx, _ in self.test_set:
            # no need to check gt
            pred_tx = self.model(rx)
            mse = tf.reduce_mean((tx - pred_tx) ** 2, keepdims=True).numpy()
            avg_mse.append(mse)

        print("[info]: <AVGMSEMetricProcessor> avg_mse_list: " + str(avg_mse))





