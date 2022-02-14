"""
Defines the Random Forest Classifier

The Random Forest classifier follows the usual algorithm for training and
classification and uses internally ID3 Trees for ensemble learning.

Wikipedia link: https://en.wikipedia.org/wiki/Random_forest
"""

import random
from math import exp, floor, sqrt

from classifier import Category, Classifier, Example
from id3 import ID3


class RandomForest(Classifier):
    """ A Random Forest classifier used to classify an Example

    ID3 Trees are used internally and the classification happens by plurality
    vote among all trees.
    """

    tree_count = 150
    examples_per_tree = 1 - 1 / exp(1)

    def __init__(self, examples: list[Example], attributes: list[str]):
        """
        Creates and trains a Random Forest classifier

        The classifier uses a number of trees, each trained on N examples
        chosen with replacement, where N is the total number of examples, and m
        attributes, chosen without replacement, where m = either sqrt(N) or M,
        where M is the total number of attributes, whichever is smaller

        :param examples: the Examples on which to train the Random Forest
        classifier
        :param attributes: the attributes that will be used to dichotomise the
        Examples on each iteration of training
        """

        self.trees: set[ID3] = set()

        example_count = floor(len(examples) * RandomForest.examples_per_tree)
        attribute_count = min(len(attributes), floor(sqrt(len(examples))))

        for _ in range(RandomForest.tree_count):
            example_subset = random.choices(examples, k=example_count)
            attribute_subset = random.sample(attributes, k=attribute_count)

            trained_tree = ID3(example_subset, attribute_subset)
            self.trees.add(trained_tree)

    def classify(self, example: Example) -> Category:
        """
        Classifies an Example by plurality vote of the individual trees

        The resulting Category of the classification is returned and, as a side
        effect, the `predicted` attribute of the Example is also updated to
        match that Category.

        :param example: the Example to classify
        :return: the predicted Category of the Example to classify
        """

        category_count: dict[Category, int] = Category.counting_dict()

        for tree in self.trees:
            category_count[tree.classify(example)] += 1

        # at this point example.predicted is whatever the last tree decided

        predicted_category = max(category_count.keys(), key=category_count.get)
        example.predicted = predicted_category
        return predicted_category
