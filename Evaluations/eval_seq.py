r"""Functional interface"""

from sklearn.metrics import precision_score, mean_absolute_error, recall_score, f1_score


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class Precision(Metric):
    """
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self):
        super(Precision, self).__init__()
        self.reset()

    def __call__(self, prediction, target, average='micro'):
        """
        :param prediction: array [ln]
        :param target: array [ln]
        :return:
        """
        self.precision = precision_score(target, prediction, average=average)

    def reset(self):
        self.precision = 0.0

    def value(self):
        return self.precision

    def name(self):
        return 'precision'


class MAE(Metric):
    """
    """

    def __init__(self):
        super(MAE, self).__init__()
        self.reset()

    def __call__(self, prediction, target):
        """
        :param prediction: array [ln]
        :param target: array [ln]
        :return:
        """
        self.mae = mean_absolute_error(target, prediction)

    def reset(self):
        self.mae = None

    def value(self):
        return self.mae

    def name(self):
        return 'mae'


class RECALL(Metric):
    """
        >>> metric = RECALL(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self, topK, ignore_index=-100):
        super(RECALL, self).__init__()
        self.topK = topK
        self.ignore_index = ignore_index
        self.reset()

    def __call__(self, predictions, target):
        """
        :param logits: array [num, topK]
        :param target: array [num]
        :return:
        """

        for i, (pred, gt) in enumerate(zip(predictions, target)):
            score = 0.0
            for rank, item in enumerate(pred):
                if item in gt:
                    score = 1.0
                    break
            self.correct_k += score
        self.total = len(target)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return 'recall'


class MRR(Metric):
    def __init__(self, topK):
        super(MRR, self).__init__()
        self.topK = topK
        self.correct = 0.0

    def __call__(self, predictions, target):
        """
        :param logits: Tensor [num, topK]
        :param target: Tesnor [num]
        :return:
        """

        for i, (pred, gt) in enumerate(zip(predictions, target)):
            score = 0.0
            for rank, item in enumerate(pred):
                if item in gt:
                    score = 1.0 / (rank + 1.0)
                    break
            self.correct += score
        self.total = len(target)

    def reset(self):
        self.correct = 0.0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'mrr'