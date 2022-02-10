import re
from enum import Enum, auto


class Category(Enum):
    """ Represents a possible Category of an Example """

    NONE = auto()
    POS = auto()
    NEG = auto()

    @classmethod
    def values(cls) -> set['Category']:
        return {cls.POS, cls.NEG}


class Example:
    """
    Represents an Example of the data. It can either be a training or a testing Example.
    The `actual` field indicates the actual Category of the Example while the `predicted`
    one should be determined upon classification. The `attributes` field contains the
    attributes of the Example, that is, the individual words in it.
    """

    _ignored_chars = ['"', "'", '.', ',', '>', '<', '\\', '/', '(', ')', ';', ':', '?']
    _regex = "[%s\\d]" % (re.escape("".join(_ignored_chars)))
    _ignored_chars_pattern = re.compile(_regex)

    def __init__(self, category: Category, raw_text: str):
        self.actual: Category = category
        self.predicted: Category = Category.NONE

        raw_attributes = raw_text.split(" ")
        self.attributes: list[str] = [Example.sanitize_attribute(attr) for attr in raw_attributes]

    @classmethod
    def sanitize_attribute(cls, attribute: str) -> str:
        return cls._ignored_chars_pattern.sub("", attribute, 0)

    @staticmethod
    def copy_of(example: 'Example') -> 'Example':
        """
        Returns a shallow copy of an Example.

        :param example: the Example to copy
        :return: a copy of the Example
        """

        copy = Example(example.actual, "")
        copy.predicted = example.predicted
        copy.attributes = example.attributes
        return copy

    def __str__(self):
        return f"{self.actual.name} - {self.predicted.name}: {self.attributes}"


class Classifier:
    """ Represents a class that can classify an Example """

    def classify(self, example: Example) -> Category:
        """
        Classifies an Example. Its `predicted` attribute is set to the
        result of the classification algorithm and is also returned

        :param example: the Example to classify
        :return: the predicted Category of the Example to classify
        """

        raise Exception("Not implemented")

    def classify_bulk(self, examples: list[Example]) -> dict[Category, int]:
        """
        Classifies the Examples and returns the number of Examples classified with each Category

        :param examples: the Examples to classify
        :return: a dictionary with the number of Examples classified with each Category
        """

        category_count = dict.fromkeys(Category.values(), 0)

        for example in examples:
            category = self.classify(example)
            category_count[category] += 1

        return category_count
