from id3 import ID3
from classifier import Example, Category, Classifier
from timed import timed

import random
import math


class RandomForest(Classifier):
    """
    A Random Forest classifier internally using ID3 trees to classify examples.
    """

    tree_count = 150
    examples_per_tree = 1 - 1 / math.exp(1)

    @classmethod
    # @timed(prompt="Train Random Forest")
    def create_timed(cls, examples: set[Example], attributes: set[str]):
        return RandomForest(examples, attributes)

    def __init__(self, examples: set[Example], attributes: set[str]):
        """
        Creates a new Random Forest classifier that uses a number of trees, each trained on a subset of the given
        attributes, to classify an Example. Note that training 5 trees each on 100% of the data or
        10 trees each on 1% of the data are both valid combinations; the product of number-of-trees times
        percentage-of-examples need not be 100%.
        :param examples: the examples on which to train the Random Forest classifier
        :param attributes: the attributes that will be used to classify the examples
        """
        # convert sets to tuples for sampling efficiency
        examples = tuple(examples)
        attributes = tuple(attributes)
        # attributes_per_tree = math.floor(math.sqrt(len(examples)))
        attributes_per_tree = min(len(attributes), math.floor(math.sqrt(len(examples))))
        examples_per_tree = math.floor(len(examples) * RandomForest.examples_per_tree)

        self.trees: set[ID3] = set()

        for _ in range(RandomForest.tree_count):
            # pass copies of the examples, so they properly hold their "predicted" value
            examples_for_tree = {Example.copy_of(e) for e in random.choices(examples, k=examples_per_tree)}
            attributes_for_tree = random.sample(attributes, k=attributes_per_tree)

            trained_tree = ID3(set(examples_for_tree), set(attributes_for_tree))
            self.trees.add(trained_tree)

    def classify(self, example: Example) -> Category:
        """
        Classifies the provided example by plurality vote of the trees created during initialization.
        :param example: The example to be classified
        :return: The predicted category of the example
        """
        category_count: dict[Category, int] = dict.fromkeys(Category.values(), 0)

        for tree in self.trees:
            category_count[tree.classify(example)] += 1

        return max(category_count.keys(), key=lambda k: category_count[k])
