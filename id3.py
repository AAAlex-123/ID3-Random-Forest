from classifier import Example, Category, Classifier
from id3_util import choose_best_attr
from timed import Timer
Priority = Timer.Priority


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

    @Timer(Priority.RUN, prompt="Train ID3")
    def __init__(self, examples: set[Example], attributes: set[str]):
        """
        Creates a new ID3 classifier by training it on the provided training data.

        :param examples: the examples on which to train the ID3 classifier
        :param attributes: the attributes that will be used to classify the examples
        cease expanding the tree
        """

        self.root: Node = self.id3_recursive(examples, attributes, Category.NONE)

    def classify(self, example_to_classify: Example) -> Category:
        """
        Classifies the provided Example by traversing the internal tree based on the
        Example's attributes. The `predicted` Category of the test_example is also updated

        :param example_to_classify: The example to be classified
        :return: The predicted Category of the example.
        """

        curr: Node = self.root
        while curr.category == Category.NONE:
            curr = curr.children[curr.attribute in example_to_classify.attributes]

        example_to_classify.predicted = curr.category
        return curr.category

    @classmethod
    def id3_recursive(cls, examples: set[Example], attributes: set[str], target_category: Category) -> Node:
        """
        Generates a tree that can classify an Example.

        :param examples: the set of Examples from which the tree will be constructed
        :param attributes: the Attributes that will be used to classify the Examples
        :param target_category: the most common category among the Examples
        :return: a tree node that best classifies the Examples with the given Attributes
        """

        # if there are no examples, return target_category
        if len(examples) == 0:
            return Node.leaf(target_category)

        # if all examples belong to a single category, return that category
        for category in Category.values():
            if all(e.actual == category for e in examples):
                return Node.leaf(category)

        # find most common category among all the examples
        categories = Category.counting_dict()
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
            if len(examples_subset) / len(examples) > cls.cutoff:
                return Node.leaf(most_common_category)

            # TODO only use attributes from the example_subset, excluding the best_attr
            attributes_subset = {a for a in attributes if a != best_attr}
            subtree = cls.id3_recursive(examples_subset, attributes_subset, most_common_category)
            root.children[value] = subtree

        return root
