import math

from classifier import Example, Category, Classifier
from timed import timed


class Node:
    """
    An internal data structure used to construct an ID3 tree and later traverse it.
    The `attribute` field indicates the attribute that is being tested at this Node and the
    `children` dictionary contains references to the appropriate Nodes that should be traversed
    based on the value of the `attribute` in an Example.
    The `category` field is NONE iff this Node is internal in the tree. A non-NONE value indicates
    that upon reaching this Node, the predicted Category of an Example will be that category.
    """

    def __init__(self, category: Category, attribute: str):
        self.category: Category = category
        self.children: dict[bool, Node] = dict()
        self.attribute: str = attribute

    @staticmethod
    def internal(attribute: str) -> 'Node':
        """
        Returns an internal Node of an ID3 tree, responsible for an attribute.
        The Node's Category is set to NONE.
        :param attribute: the attribute that shall be checked in this Node
        :return: the Node
        """
        return Node(Category.NONE, attribute)

    @staticmethod
    def leaf(category: Category) -> 'Node':
        """
        Returns a leaf Node of an ID3 tree, responsible for a classification.
        The Node's Attribute is set to "".
        :param category: the Category with which Examples will be classified according to this Node
        :return: the Node
        """
        return Node(category, "")


class ID3(Classifier):
    """ An ID3 Tree classifier used to classify an Example """

    cutoff = 0.95

    @classmethod
    # @timed(prompt="Train ID3")
    def create_timed(cls, examples: set[Example], attributes: set[str]):
        """
        A timed wrapper for the creation of an ID3 tree.
        :param examples: the examples on which to train the ID3 classifier
        :param attributes: the attributes that will be used to classify the examples
        cease expanding the tree
        :return: a new ID3 tree
        """
        return ID3(examples, attributes)

    def __init__(self, examples: set[Example], attributes: set[str]):
        """
        Creates a new ID3 classifier by training it on the provided training data.
        :param examples: the examples on which to train the ID3 classifier
        :param attributes: the attributes that will be used to classify the examples
        cease expanding the tree
        """
        self.root: Node = self.id3_recursive(examples, attributes, Category.NONE)

    def classify(self, test_example: Example) -> Category:
        """
        Classifies the provided Example by traversing the internal tree based on the
        Example's attributes.The `predicted` Category of the test_example is also updated
        :param test_example: The example to be classified
        :return: The predicted Category of the example.
        """
        curr: Node = self.root
        while curr.category == Category.NONE:
            curr = curr.children[curr.attribute in test_example.attributes]

        test_example.predicted = curr.category
        return curr.category

    def id3_recursive(self, examples: set[Example], attributes: set[str], target_category: Category) -> Node:
        """
        Generates a tree that can classify an example.
        :param examples: the set of examples from which the tree will be constructed
        :param attributes: the attributes that will be used to classify the examples
        :param target_category: the most common category among the examples
        :return: a tree node that best classifies the examples with the given attributes
        """
        # if there are no examples, return target_category
        if len(examples) == 0:
            return Node.leaf(target_category)

        # if all examples belong to a single category, return that category
        for category in Category.values():
            if all(e.actual == category for e in examples):
                return Node.leaf(category)

        # find most common category among all the examples
        categories = dict.fromkeys(Category.values(), 0)
        for example in examples:
            categories[example.actual] += 1
        most_common_category = max(categories.keys(), key=lambda k: categories[k])

        # if there are no attributes left, return the most common category
        if len(attributes) == 0:
            return Node.leaf(most_common_category)

        # otherwise, create a tree by splitting the examples by whether
        # they contain best_attr or not (values True or False)
        best_attr = choose_best_attr(attributes, examples)
        root = Node.internal(best_attr)

        for value in {True, False}:
            examples_subset = {e for e in examples if (best_attr in e.attributes) == value}

            # if sufficient categorization, end the tree expansion early
            if len(examples_subset) / len(examples) > ID3.cutoff:
                return Node.leaf(most_common_category)

            attributes_subset = {a for a in attributes if a != best_attr}
            subtree = self.id3_recursive(examples_subset, attributes_subset, most_common_category)
            root.children[value] = subtree

        return root


def choose_best_attr(attributes: set[str], examples: set[Example]) -> str:
    """
    Returns the attribute with the maximum information gain calculated for a set of Examples.
    :param attributes: the attributes to be examined for information gain
    :param examples: the examples for which the information gain will be calculated
    :return: the attribute with the maximum information gain for the examples given
    """

    # find the word with the maximum information gain
    max_gain = -1
    max_gain_word = ""
    for attribute in attributes:
        gain = info_gain(examples, attribute)
        if gain > max_gain:
            max_gain = gain
            max_gain_word = attribute

    return max_gain_word


def info_gain(examples: set[Example], attribute: str) -> float:
    """
    Returns the information gain of an attribute in a set of Examples.
    :param examples: the set of examples among which to calculate the information gain
    :param attribute: the attribute for which to calculate the information gain
    :return: the information gain of that word in the examples
    """
    example_count = len(examples)
    pos_examples = 0
    x_count = x_count_in_pos = x_count_in_neg = 0

    for example in examples:
        if example.actual == Category.POS:
            pos_examples += 1

        if attribute in example.attributes:
            x_count += 1
            if example.actual == Category.POS:
                x_count_in_pos += 1
            elif example.actual == Category.NEG:
                x_count_in_neg += 1

    # H(C) - P(X=1) * H(C|X=1) + P(X=0) * H(C|X=0)
    hc = entropy(pos_examples / example_count)
    px1 = x_count / example_count
    pc1x1 = 0.0 if x_count == 0 else x_count_in_pos / x_count
    hcx1 = entropy(pc1x1)
    px0 = 1 - px1
    pc1x0 = 0.0 if x_count == 0 else x_count_in_neg / x_count
    hcx0 = entropy(pc1x0)

    return hc - (px1 * hcx1 + px0 * hcx0)


def entropy(probability: float) -> float:
    """
    Returns the entropy associated with the probability of an event.
    :param probability: the probability of the event being true
    :return: the entropy of that event
    """

    if probability == 0.0 or probability == 1.0:
        return 0.0

    p_true = probability
    p_false = 1 - probability
    return - p_true * math.log2(p_true) - p_false * math.log2(p_false)
