"""
Defines the ID3 Classifier

The ID3 classifier follows the usual algorithm for training and classification
and uses internally the Node class for the nodes of the tree.

Wikipedia link: https://en.wikipedia.org/wiki/ID3_algorithm
"""

from classifier import Category, Classifier, Example
from id3_util import find_best_attribute


class Node:
    """
    Represents a Node of an ID3 tree

    It is used to construct (train) and then traverse an ID3 tree, in order to
    classify an Example. The `attribute` field indicates the attribute that is
    being tested at this Node and the `children` dictionary points to the Node
    that should be traversed next based on the value of the `attribute` in the
    Example that is being classified.

    The `category` field is NONE iff this Node is internal in the tree, that is
    no classification has been yet determined for the Example. A non-NONE value
    indicates that upon reaching this Node the predicted Category of the
    Example is known to be that category.

    It is advised to use the provided static methods to initialise internal and
    leaf Nodes for better code readability.
    """

    def __init__(self, category: Category, attribute: str):
        self.category: Category = category
        self.children: dict[bool, Node] = dict.fromkeys({True, False})
        self.attribute: str = attribute

    @staticmethod
    def internal(attribute: str) -> 'Node':
        """
        Returns an internal Node of an ID3 tree, a Node which cannot yet
        predict the Category of an Example but instead points to the other
        Nodes of the tree based on an attribute. Its `actual` field is set to
        the NONE Category.

        :param attribute: the attribute that will be checked in this Node
        :return: the Node
        """
        return Node(Category.NONE, attribute)

    @staticmethod
    def leaf(category: Category) -> 'Node':
        """
        Returns a leaf Node of an ID3 tree, a Node which can predict the
        Category of an Example. Its `attribute` field is set to "".

        :param category: the Category with which Examples will be classified
        :return: the Node
        """
        return Node(category, "")


class ID3(Classifier):
    """ An ID3 Tree classifier used to classify an Example """

    cutoff = 0.95

    def __init__(self, examples: list[Example], attributes: list[str]):
        """
        Constructs an ID3 classifier and trains it on the provided Examples

        :param examples: the Examples on which to train the ID3 classifier
        :param attributes: the attributes that will be used to dichotomise the
        Examples on each iteration of training
        """

        self.root = self.id3_recursive(examples, attributes)

    def classify(self, example: Example) -> Category:
        """
        Classifies the provided Example by traversing the internal tree based
        on the Example's attributes. The `predicted` Category of it is updated.

        :param example: the Example to be classified
        :return: the predicted Category of the Example.
        """

        curr = self.root
        while curr.category == Category.NONE:
            curr = curr.children[curr.attribute in example.attributes]

        example.predicted = curr.category
        return curr.category

    @classmethod
    def id3_recursive(cls, examples: list[Example], attributes: list[str]) -> \
            Node:
        """
        Generates a tree that can classify an Example using the ID3 algorithm

        :param examples: the set of Examples that will be dichotomised to
        construct the Nodes of the tree
        :param attributes: the attributes that will determine how to
        dichotomise the Examples on each iteration
        :return: a Node whose tree best classifies the Examples
        """

        # if all examples belong to a single category, return that category
        for category in Category.values():
            if all(example.actual == category for example in examples):
                return Node.leaf(category)

        # find most common category among all the examples
        categories = Category.counting_dict()
        for example in examples:
            categories[example.actual] += 1
        most_common_category = max(categories.keys(), key=categories.get)

        # if there are no attributes left, return the most common category
        if len(attributes) == 0:
            return Node.leaf(most_common_category)

        # otherwise, create a tree by splitting the examples by whether they
        # contain best_attr or not (values True or False)
        best_attr = find_best_attribute(attributes, examples)
        root = Node.internal(best_attr)

        def predicate(expected: bool):
            def inner(training_example: Example) -> bool:
                return (best_attr in training_example.attributes) == expected
            return inner

        for value in {True, False}:
            example_subset = list(filter(predicate(value), examples))

            # if no examples with this value exist, return most common category
            if len(example_subset) == 0:
                root.children[value] = Node.leaf(most_common_category)
                continue

            # NOTE: this `if` doesn't make sense if there are only two values:
            # if the above `if` executes, then this one will for sure. But, for
            # more than two values, this `if` is not dependent from the above

            # if sufficient categorization, end the tree expansion early
            if len(example_subset) / len(examples) > cls.cutoff:
                return Node.leaf(most_common_category)

            # TODO only use attributes from the examples, excluding best_attr
            # but is it a good idea???
            attribute_subset = [a for a in attributes if a != best_attr]
            subtree = cls.id3_recursive(example_subset, attribute_subset)
            root.children[value] = subtree

        return root
