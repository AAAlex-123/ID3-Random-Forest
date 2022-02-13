"""
Defines the classes that will be used by the Classifiers

Category: a Category where an Example belongs
Example: an Example that is associated with data and has a Category
Classifier: superclass for all classifiers that can be trained to classify
        Examples
"""

from enum import Enum, auto


class Category(Enum):
    """ Represents a possible Category of an Example """

    NONE = auto()
    POS = auto()
    NEG = auto()

    @classmethod
    def values(cls) -> list['Category']:
        """ Returns all the Categories except NONE

        :return: a list with the Categories
        """
        return [c for c in iter(cls) if c != cls.NONE]

    @classmethod
    def counting_dict(cls, value: int = 0) -> dict['Category', int]:
        """ Returns a Category-int dictionary

        :param value: the initial value for each key in the dictionary
        :return: the dictionary
        """
        return dict.fromkeys(cls.values(), value)


class Example:
    """
    Represents an Example of the data

    The `actual` field indicates the actual Category of the Example, which is
    determined when it is created, while the `predicted` field is determined
    upon classification. The `attributes` field contains the attributes of the
    Example, that is, its individual words.
    """

    import re

    __ignored_chars = ['"', "'", '.', ',', '>', '<', '\\', '/', '(', ')', ';',
                       ':', '?']
    __regex = "[%s\\d]" % (re.escape("".join(__ignored_chars)))
    __ignored_chars_pattern = re.compile(__regex)

    def __init__(self, category: Category, raw_text: str) -> None:
        """
        Initialises an Example

        Its attributes are extracted and sanitized from the raw_text.

        :param category: the actual Category of this Example
        :param raw_text: the text from which to extract the attributes
        """

        self.actual: Category = category
        self.predicted: Category = Category.NONE

        raw_attributes = raw_text.split(" ")
        clean_attributes = map(Example.sanitize_attribute, raw_attributes)
        self.attributes: list[str] = list(clean_attributes)

    @classmethod
    def sanitize_attribute(cls, attribute: str) -> str:
        """
        Sanitises an attribute by removing all ignored characters

        :param attribute: the attribute to sanitise
        :return: the sanitised attribute
        """
        return cls.__ignored_chars_pattern.sub("", attribute, 0)

    @staticmethod
    def copy_of(example: 'Example') -> 'Example':
        """
        Returns a shallow copy of an Example

        :param example: the Example to copy
        :return: a shallow copy of the Example
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
        Classifies an Example

        The resulting Category of the classification is returned and, as a side
        effect, the `predicted` attribute of the Example is also updated to
        match that Category.

        Subclasses need to override this method accordingly.

        :param example: the Example to classify
        :return: the predicted Category of the Example to classify
        """

        raise NotImplementedError("Classifier.classify(Example)")

    def classify_bulk(self, examples: list[Example]) -> dict[Category, int]:
        """
        Classifies a collection of Examples

        The `predicted` attribute of each Example is updated with the result of
        its classification and additionally the number of Examples classified
        with each Category is counted and returned as a dictionary.

        :param examples: the Examples to classify
        :return: a dictionary containing the number of Examples classified with
        each Category
        """

        category_count = Category.counting_dict()

        for example in examples:
            category = self.classify(example)
            category_count[category] += 1

        return category_count
