""" Defines utility functions for the training of the ID3 classifier """

import math

from classifier import Category, Example


def find_best_attribute(attributes: list[str], examples: list[Example]) -> str:
    """
    Returns the attribute with the maximum information gain.

    :param attributes: the attributes to be examined for information gain
    :param examples: the Examples for which the information gain will be
    calculated
    :return: the Attribute with the maximum information
    """

    return max(attributes, key=_information_gain_calculator(examples))


def _information_gain_calculator(examples: list[Example]):
    """
    A curried version of compute_information_gain with a given list of Examples

    Returns a function that computes the information gain for an attribute

    :param examples: the Examples among which to calculate the information gain
    :return: the function
    """

    def inner(attribute: str) -> float:
        return _compute_info_gain(attribute, examples)

    return inner


def _compute_info_gain(attribute: str, examples: list[Example]) -> float:
    """
    Returns the information gain of an attribute calculated for some Examples.

    :param examples: the Examples among which to calculate the information gain
    :param attribute: the attribute for which to calculate the information gain
    :return: the information gain of that attribute in the Examples
    """

    example_count_per_category = Category.counting_dict()
    attribute_count_per_category = Category.counting_dict()

    for example in examples:
        example_count_per_category[example.actual] += 1

        if attribute in example.attributes:
            attribute_count_per_category[example.actual] += 1

    example_count = len(example_count_per_category)
    attribute_count = len(attribute_count_per_category)
    pos_count = attribute_count_per_category[Category.POS]
    neg_count = attribute_count_per_category[Category.NEG]

    # H(C) - (P(X=1) * H(C|X=1) + P(X=0) * H(C|X=0))
    hc = _entropy(example_count_per_category[Category.POS] / example_count)

    px1 = attribute_count / example_count
    pc1x1 = 0.0 if attribute_count == 0 else pos_count / attribute_count
    hcx1 = _entropy(pc1x1)

    px0 = 1 - px1
    pc1x0 = 0.0 if attribute_count == 0 else neg_count / attribute_count
    hcx0 = _entropy(pc1x0)

    return hc - (px1 * hcx1 + px0 * hcx0)


def _entropy(probability: float) -> float:
    """
    Returns the entropy associated with the probability of an event

    :param probability: the probability of the event being True
    :return: the entropy of that event
    """

    if probability == 0.0 or probability == 1.0:
        return 0.0

    p_true = probability
    p_false = 1 - probability
    return - p_true * math.log2(p_true) - p_false * math.log2(p_false)
