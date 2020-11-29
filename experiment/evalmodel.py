import sys

sys.path.append('../utils')
sys.path.append('../data')

from enum import Enum
import tensorflow as tf
from dataset import DataSetV1, DataSetV2
from plotutils import PlotUtils

'''
@author: Chen Hao
@filename: evalmodel.py
@commment: this file implements a chained modelEvaluator, the user can add metric processor to the 
modelEvaluator and the modelEvaluator will automatically process the evaluation-task chain depending on the 
metric to evaluate.
@date: 2020-09-23
'''


class Metric(Enum):
    """
    enum metrics for the ModelEvaluator
    """
    # bit error rate, noting that the predicted signal is not passed into a softmax classifier
    BER = 0

    # average mse in the prediction, noting that the predicted signal is before decision
    AVG_MSE = 1

    # bit error rate, directly using the classifier to do symbol decision
    BER_SOFTMAX = 2

    # BER REG
    BER_REGRESSION = 3


class ModelEvaluator:
    """
    Model Evaluator class is used to eval a trained nn model using several test set on different metrics.
    """

    def __init__(self, model: tf.keras.Model, dataset):

        """
        :param model: tf model
        :param dataset: dataset to eval
        """
        # model to evaluate
        self.model = model

        # test dataset, test one b
        self.testset = dataset

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
        elif metric == Metric.BER_SOFTMAX:
            self.metric_processor_list.append(
                BERSoftmaxMetricProcessor(Metric.BER_SOFTMAX.value, self.model, self.testset))
        elif metric == Metric.BER_REGRESSION:
            self.metric_processor_list.append(BERRegressionMetricProcessor(Metric.BER_REGRESSION.value, self.model, self.testset))

        return self

    def do_eval(self):
        if len(self.metric_processor_list) == 0:
            print("[Warning]: the {do_eval} function is called, "
                  "but no metric processor is added to the model evaluator")
            return
        else:
            print("\033[1;32m" + "[info]: (ModelEvaluator) evaluating..." + " \033[0m")
            for processor in self.metric_processor_list:
                # directly call the processor's process method, since the evaluation context has been built before.
                processor.process()


class MetricProcessor:
    """
    base class of metric processor for evaluating different metrics
    """

    def __init__(self, metric_type: int, model: tf.keras.Model, test_set):
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
    ber metric processor, should pass DataSetV2 to do the evaluation
    """

    def __init__(self, metric_type: int, model: tf.keras.Model, test_set: DataSetV2):
        # directly call the super class to init the test context.
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the test dataset and calculate the ber metric
        :return: the BER performance of the model
        """
        right = 0
        error = 0
        for tx, rx, gt in self.test_set:
            pred = self.model(rx)
            res = self._decode_pam4(pred)
            res_map = [int(gt.numpy()[i]) == res[i] for i in range(len(gt))]
            right_num = sum(res_map)
            error_num = len(res_map) - right_num

            # show why error happens
            # if error_num != 0:
            #     tx = tf.squeeze(tx, axis=0)
            #     rx = tf.squeeze(rx, axis=0)
            #     pred = tf.squeeze(pred, axis=0)
            #     PlotUtils.plot_wave_tensors(tx, rx, pred, legend_list=['tx', 'rx', 'clean_wave'])

            # ber statistics
            right += right_num
            error += error_num

        print("[info]: <BERMetricProcessor> total symbol: " + str(right + error) + " , right decision: " +
              str(right) + " ,error decision: " + str(error) + ", ber: " + str(error / (right + error)))

    def _decode_pam4(self, pred_tx):
        """
        function used to decode the signal in sig_range (default (-1, 1)) to standard pam4 signal (-3, -1, 1, 3)
        :param pred_tx: tf.Tensor, basically the predicted result of the cleaner
        :return: a list of the decoding results, the length is same with the batch size of pred_tx
        """
        batch_sz = pred_tx.shape[0]
        sample_index = ((self.test_set.sym_win_size - 1) * self.test_set.sample_per_symbol) // 2
        res = list()
        for i in range(batch_sz):
            ds = float(pred_tx[i, int(sample_index)].numpy())

            if ds > 0.4119:
                de = 3
            elif 0.4119 >= ds > 0:
                de = 1
            elif 0 >= ds > -0.4119:
                de = -1
            else:
                de = -3

            res.append(de)

        return res


class AVGMSEMetricProcessor(MetricProcessor):
    """
    AVG_MSE metric processor
    """

    def __init__(self, metric_type: int, model: tf.keras.Model, test_set):
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the average accuracy on the test dataset
        :return:
        """
        inc = 0
        mse = 0
        for tx, rx, _ in self.test_set:
            # no need to check gt
            inc += 1
            pred_tx = self.model(rx)
            mse += tf.reduce_mean((tx - pred_tx) ** 2, keepdims=True).numpy()

        mse /= inc
        print("[info]: <AVGMSEMetricProcessor> avg_mse: " + str(mse))


class BERRegressionMetricProcessor(MetricProcessor):
    """
    BER Regression processor
    """

    def __init__(self, metric_type: int, model: tf.keras.Model, test_set):
        """

        :param metric_type:
        :param model:
        :param test_set: should be TestDataSetV2
        """
        if not isinstance(test_set, DataSetV1):
            raise TypeError("[Error]: In BERSoftmaxMetricProcessor, the test dataset should be the type {"
                            "DataSetV1}, "
                            " but got ", str(type(test_set)))
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the test dataset and calculate the ber metric
        :return: the BER performance of the model
        """
        right = 0
        error = 0
        for tx, rx, gt in self.test_set:
            pred = self.model(rx)
            sample_index = ((self.test_set.sym_win_size - 1) * self.test_set.sample_per_symbol) // 2
            gt_pred = tx[0, sample_index]
            res = self._decode_pam4(pred)

            res_map = [int(gt.numpy()[i]) == res[i] for i in range(len(gt))]
            right_num = sum(res_map)
            error_num = len(res_map) - right_num

            # show why error happens
            # if error_num != 0:
            #     tx = tf.squeeze(tx, axis=0)
            #     rx = tf.squeeze(rx, axis=0)
            #     pred = tf.squeeze(pred, axis=0)
            #     PlotUtils.plot_wave_tensors(tx, rx, pred, legend_list=['tx', 'rx', 'clean_wave'])

            # ber statistics
            right += right_num
            error += error_num

        print("[info]: <BERMetricProcessor> total symbol: " + str(right + error) + " , right decision: " +
              str(right) + " ,error decision: " + str(error) + ", ber: " + str(error / (right + error)))

    def _decode_pam4(self, pred_tx):
        """
        function used to decode the signal in sig_range (default (-1, 1)) to standard pam4 signal (-3, -1, 1, 3)
        :param pred_tx: tf.Tensor, basically the predicted result of the cleaner
        :return: a list of the decoding results, the length is same with the batch size of pred_tx
        """
        batch_sz = pred_tx.shape[0]
        sample_index = 0
        res = list()
        for i in range(batch_sz):
            ds = float(pred_tx[i, int(sample_index)].numpy())

            if ds > 0.4119:
                de = 3
            elif 0.4119 >= ds > 0:
                de = 1
            elif 0 >= ds > -0.4119:
                de = -1
            else:
                de = -3

            res.append(de)

        return res


class BERSoftmaxMetricProcessor(MetricProcessor):
    """
    BER Softmax Metric Processor
    """

    def __init__(self, metric_type: int, model: tf.keras.Model, test_set):
        """

        :param metric_type:
        :param model:
        :param test_set: should be TestDataSetV2
        """
        if not isinstance(test_set, DataSetV2):
            raise TypeError("[Error]: In BERSoftmaxMetricProcessor, the test dataset should be the type {"
                            "DataSetV2}, "
                            " but got ", str(type(test_set)))
        super().__init__(metric_type, model, test_set)

    def process(self):
        """
        do process the ber metric with the softmax result, the code should be run in eager mode.
        :return: None
        """
        total = 0
        right = 0
        error = 0
        for _, rx, gt in self.test_set:
            total += 1
            pred = self.model(rx)

            # argmax to find which one is in most probability
            pred = tf.argmax(tf.squeeze(pred, axis=0)).numpy()
            gt = tf.argmax(tf.squeeze(gt, axis=0).numpy()).numpy()
            if pred == int(gt):
                right += 1
            else:
                error += 1

        print("[info]: <BERSoftmaxMetricProcessor> total symbol: " + str(right + error) + " , right decision: " +
              str(right) + " ,error decision: " + str(error) + ", ber: " + str(error / (right + error)))
