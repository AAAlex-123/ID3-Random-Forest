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
        self.attributes: set[str] = {Example.sanitize_attribute(attr) for attr in raw_attributes}

    @classmethod
    def sanitize_attribute(cls, attribute: str) -> str:
        return cls._ignored_chars_pattern.sub("", attribute, 0)

    def copy(self):
        """
        Create a new Example with the same attributes (shallow copy) and actual category
        but with a blank predicted category.
        """
        ex = Example(self.actual, "")
        ex.attributes = self.attributes
        return ex

    def __str__(self):
        return f"{self.actual.name} - {self.predicted.name}: {self.attributes}"


class Classifier:

    def classify(self, example: Example) -> Category:
        pass

    def classify_bulk(self, examples: tuple[Example]) -> None:
        for example in examples:
            self.classify(example)
