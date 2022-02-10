from classifier import Example, Category


class ClassifierEvaluation:
    def __init__(self, classified_examples: set[Example], b: float = 1):

        example_count = len(classified_examples)
        true_positives = true_negatives = false_negatives = false_positives = 0

        for example in classified_examples:
            if example.actual == Category.POS and example.predicted == Category.POS:
                true_positives += 1
            elif example.actual == Category.NEG and example.predicted == Category.NEG:
                true_negatives += 1
            elif example.actual == Category.POS and example.predicted == Category.NEG:
                false_negatives += 1
            elif example.actual == Category.NEG and example.predicted == Category.POS:
                false_positives += 1

        self.accuracy_true: float = (true_positives + true_negatives) / example_count

        self.precision_pos: float = true_positives / (true_positives + false_positives)
        self.precision_neg: float = true_negatives / (true_negatives + false_negatives)

        self.recall_pos: float = true_positives / (true_positives + false_negatives)
        self.recall_neg: float = true_negatives / (true_negatives + false_positives)

        self.f_measure_pos: float = ((b * b + 1) * self.precision_pos * self.recall_pos) / (b * b * self.precision_pos + self.recall_pos)
        self.f_measure_neg: float = ((b * b + 1) * self.precision_neg * self.recall_neg) / (b * b * self.precision_neg + self.recall_neg)

        self.macro_precision = 1 / 2 * (self.precision_pos + self.precision_neg)
        self.macro_recall = 1 / 2 * (self.recall_pos + self.recall_neg)

        self.micro_precision = None
        self.micro_recall = None

        self.sensitivity = self.recall_pos
        self.specificity = self.recall_neg
        self.true_positive_rate = self.sensitivity
        self.false_positive_rate = 1 - self.specificity

    def __str__(self):
        return f"Accuracy: {self.accuracy_true}\nPrecision: {self.precision_pos}\n" \
               f"Recall: {self.recall_pos}\nF Measure (b=0.5): {self.f_measure_pos}\n"
