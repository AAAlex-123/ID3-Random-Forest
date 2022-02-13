import math

from classifier import Example, Category
from timed import Timer
Priority = Timer.Priority


@Timer(priority=Priority.DEBUG)
def choose_best_attr(attributes: set[str], examples: set[Example]) -> str:
    """
    Returns the attribute with the maximum information gain calculated for a set of Examples.

    :param attributes: the Attributes to be examined for information gain
    :param examples: the Examples for which the information gain will be calculated
    :return: the Attribute with the maximum information gain for the Examples given
    """

    return max(attributes, key=lambda attr: calculate_information_gain(attr, examples))


def calculate_information_gain(attribute: str, examples: set[Example]) -> float:
    """
    Returns the information gain of an Attribute in a set of Examples.

    :param examples: the Examples among which to calculate the information gain
    :param attribute: the Attribute for which to calculate the information gain
    :return: the information gain of that Attribute in the Examples
    """

    example_count_per_category = Category.counting_dict()
    attribute_count_per_category = Category.counting_dict()

    for example in examples:
        example_count_per_category[example.actual] += 1

        if attribute in example.attributes:
            attribute_count_per_category[example.actual] += 1

    example_count = len(example_count_per_category)
    attribute_count = len(attribute_count_per_category)

    # H(C) - (P(X=1) * H(C|X=1) + P(X=0) * H(C|X=0))
    hc = entropy(example_count_per_category[Category.POS] / example_count)

    px1 = attribute_count / example_count
    pc1x1 = 0.0 if attribute_count == 0 else attribute_count_per_category[Category.POS] / attribute_count
    hcx1 = entropy(pc1x1)

    px0 = 1 - px1
    pc1x0 = 0.0 if attribute_count == 0 else attribute_count_per_category[Category.NEG] / attribute_count
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
