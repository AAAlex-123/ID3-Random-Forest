import math
import random

from id3 import ID3
from classifier import Example, Category, Classifier
from timed import Timer
Priority = Timer.Priority


class RandomForest(Classifier):
    """ A Random Forest classifier that internally uses ID3 trees to classify Examples """

    tree_count = 150
    examples_per_tree = 1 - 1 / math.exp(1)

    @Timer(priority=Priority.RUN, prompt="Train Random Forest")
    def __init__(self, examples: set[Example], attributes: set[str]):
        """
        TODO set to tuples

        Creates a new Random Forest classifier that uses a number of trees, each trained on a subset of the given
        attributes, to classify an Example.

        :param examples: the Examples with which to train this Random Forest classifier
        :param attributes: the Attributes that will be used to classify the Examples
        """

        # convert sets to tuples for sampling efficiency
        examples = tuple(examples)
        attributes = tuple(attributes)

        examples_per_tree = math.floor(len(examples) * RandomForest.examples_per_tree)
        attributes_per_tree = min(len(attributes), math.floor(math.sqrt(len(examples))))

        self.trees: set[ID3] = set()

        for _ in range(RandomForest.tree_count):
            # pass copies of the examples, so they properly hold their "predicted" value
            examples_for_tree = {Example.copy_of(e) for e in random.choices(examples, k=examples_per_tree)}
            attributes_for_tree = random.sample(attributes, k=attributes_per_tree)

            trained_tree = ID3(set(examples_for_tree), set(attributes_for_tree))
            self.trees.add(trained_tree)

    def classify(self, example: Example) -> Category:
        """
        Classifies an Example by plurality vote of the individual trees.

        :param example: The Example to be classified
        :return: The predicted category of the Example
        """

        category_count: dict[Category, int] = Category.counting_dict()

        for tree in self.trees:
            category_count[tree.classify(example)] += 1

        return max(category_count.keys(), key=lambda k: category_count[k])
