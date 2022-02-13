from classifier import Example, Category


class ClassifierEvaluation:
    """ TODO """

    def __init__(self, classified_examples: set[Example]):

        self._example_count = len(classified_examples)
        self._data = {}
        for category in Category.values():
            self._data[category] = dict.fromkeys({True, False}, 0)

        for category in Category.values():
            for example in classified_examples:
                if example.predicted == category:
                    self._data[category][example.actual == category] += 1

    def accuracy(self) -> float:
        if self._example_count == 0:
            return 0

        return sum(self._data[c][True] for c in Category.values()) / self._example_count

    def precision(self, category: Category) -> float:
        predicted_of_category = self._data[category][True]
        total_classified_of_category = self._data[category][True] + self._data[category][False]

        if total_classified_of_category == 0:
            return 1

        return predicted_of_category / total_classified_of_category

    def recall(self, category: Category) -> float:
        predicted_of_category = self._data[category][True]
        true_members_of_category = self._data[category][True] + sum(self._data[c][False] for c in Category.values() if c != category)

        if true_members_of_category == 0:
            return 0

        return predicted_of_category / true_members_of_category

    def f_measure(self, category: Category, b: float) -> float:
        precision = self.precision(category)
        recall = self.recall(category)

        if precision == 0 and recall == 0:
            return 0

        if b == 0 and recall == 0:
            return 0

        return ((b * b + 1) * precision * recall) / (b * b * precision + recall)

    def macro_precision(self) -> float:
        return sum(self.precision(c) for c in Category.values()) / len(self._data)

    def macro_recall(self) -> float:
        return sum(self.recall(c) for c in Category.values()) / len(self._data)

    def micro_precision(self) -> float:
        enumerator = sum(self._data[c][True] for c in Category.values())
        denominator = sum(self._data[c][True] + self._data[c][False] for c in Category.values())
        return enumerator / denominator

    def micro_recall(self) -> float:
        enumerator = sum(self._data[c][True] for c in Category.values())
        denominator = sum(self._data[c][True] + sum(self._data[c1][False] for c1 in Category.values() if c1 != c)
                          for c in Category.values())
        return enumerator / denominator

    def basic_stats(self) -> str:
        return f"Accuracy: {self.accuracy()}\nPrecision: {self.precision(Category.POS)}\n" \
               f"Recall: {self.recall(Category.POS)}\nF Measure (b=1): {self.f_measure(Category.POS, b=1)}\n"
