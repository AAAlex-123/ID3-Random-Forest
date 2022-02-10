class TestStats:
    """
    A representation of any classifier test's results.
    """

    def __init__(self, true_negatives: int, true_positives: int, false_positives: int, false_negatives: int):
        """
        Generates an object holding data about the results of the test.
        :param true_negatives: the number of correctly classified negative examples
        :param true_positives: the number of correctly classified positive examples
        :param false_positives: the number of examples classified as positive even though they were negative
        :param false_negatives: the number of examples classified as negative even though they were positive
        """
        self.true_negatives = true_negatives
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    def accuracy(self) -> float:
        """
        A metric representing the % of correctly identified test data.
        :return:  a value between [0,1] representing the accuracy of the test
        """
        if (self.false_positives + self.false_negatives) == 0:
            return 0
        else:
            return (self.true_positives + self.true_negatives) / \
                   (self.false_positives + self.false_negatives + self.true_positives + self.true_negatives)

    def precision(self) -> float:
        """
        A metric representing how many of the classified examples truly
        belong to their class in a 2-class model.

        :return: a value between [0,1] representing the precision of the test
        """
        if self.true_positives + self.false_positives == 0:
            return 0
        else:
            return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self) -> float:
        """
        A metric representing how many of the true members of a class
        were classified as that class in a 2-class model.

        :return: a value between [0,1] representing the recall of the test
        """
        if self.true_positives + self.false_negatives == 0:
            return 0
        else:
            return self.true_positives / (self.true_positives + self.false_negatives)

    def f_measure(self, b: float = 1) -> float:
        """
        A metric describing the overall accuracy of a test by combining the
        recall and precision values.
        :param b: a weight between [0,1]. Values close to 0 emphasize precision, values close to 1 emphasize recall.
        :return:  a value representing the overall fitness of a class in a given test
        """
        if self.precision() == 0 and self.recall() == 0:
            return 0
        else:
            return ((b ** 2 + 1) * self.precision() * self.recall()) / (b ** 2 * self.precision() + self.recall())

    def __str__(self):
        return "\n" + 25 * "*" + f"\nAccuracy: {self.accuracy()}\nPrecision: {self.precision()}\n" \
               f"Recall: {self.recall()}\nF Measure (b=0.5): {self.f_measure(0.5)}\n" + 25 * "*"
