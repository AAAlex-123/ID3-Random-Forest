import os

from file_util import reader
from id3 import Category, Example


def load_examples_from_directory(directory: str, categories: set[Category],
                                 sample_size: int) -> list[Example]:
    """ Loads a number of Examples from some Categories

    The sample size is split evenly among the different Categories. Each
    Category is expected to be located in a different directory whose name is
    the name of the Category in lowercase (category.name.lower()).

    :param directory: the top-level directory containing the data
    :param categories: the Categories for which to load the Examples
    :param sample_size: the total number of Examples to load, it is split
    evenly among all Categories
    :return: a list of all the Examples that were loaded
    """

    examples = list()
    examples_per_category = sample_size // len(categories)

    for category in categories:
        directory_for_category = os.path.join(directory, category.name.lower())
        new_examples = _load_examples_of_category(directory_for_category,
                                                  category,
                                                  examples_per_category)
        examples.extend(new_examples)

    return examples


def _load_examples_of_category(directory: str, category: Category,
                               sample_size: int, verbose: bool = False) \
        -> list[Example]:
    """ Loads a number of Examples from a Category

    The Examples are loaded from the directory and their `actual` field is set
    to the given Category.

    :param directory: the directory containing the data
    :param category: the Category of the data. Each Example will have this as
    its `actual` field
    :param sample_size: the maximum number of Examples to load
    :param verbose: whether to print progress information
    :return: a list with the loaded Examples
    """

    files = os.listdir(directory)
    data: list[str] = list()
    one_tenth_progress = min(len(files), sample_size) // 10

    for count, file in enumerate(files):
        if count == sample_size:
            break

        with reader(os.path.join(directory, file)) as f:
            contents = f.read().lower()

        data.append(contents)

        if verbose and (count + 1) % one_tenth_progress == 0:
            percent_done = (count + 1) // one_tenth_progress * 10
            print(f"\r{percent_done}% complete...", end="", flush=True)

    if verbose:
        print()  # move cursor to the next line because of end="" 3 lines above

    return [Example(category, text) for text in data]


def load_attributes_from_file(filename: str, count: int, ignore: int) \
        -> list[str]:
    """ Loads a number of attributes from a file

    The attributes are loaded and returned in order of appearance in the file.
    If there are less than `ignore` attributes in the file, an empty list is
    returned. If the file contains fewer than `ignore` + `count` attributes,
    then `ignore` attributes are still ignored but fewer than `count` are
    returned

    :param filename: the name of the file containing the attributes
    :param count: the number of attributes to load
    :param ignore: the number of attributes to ignore at the start
    :return: a list with the
    """

    raw_attributes = list()

    with reader(filename) as f:
        for _ in range(ignore):
            if (ignored_attribute := f.readline()) == "":
                return []

        for _ in range(count):
            if (attribute := f.readline()) == "":
                break
            raw_attributes.append(attribute[:-1])

    print(f"TODO: LOAD_IMDB - LOAD_ATTRIBUTES_FROM_FILE - REMOVE\n{raw_attributes}")
    return [Example.sanitize_attribute(attr) for attr in raw_attributes]


def filter_attributes_by_examples(attributes: list[str],
                                  examples: list[Example]) -> list[str]:
    # TODO figure out what the fuck happens when attribute that does not
    # appear in the Examples is included

    # the find_best_attribute in id3 what should return?
    # what is the information gain of an attribute that isn't present?
    # is the HC correct in id3_util?
    # is entropy of 0 correctly returned?
    # if the big formula for info gain correct? why hc - (stuff) where stuff is
    # 0 when attribute is not present in any of the examples?

    """ Filters attributes by whether they are present in the Examples

    The list is filtered and only attributes that are present in at least one
    Example are retained. This removes any attributes that don't provide any
    actual information. TODO: what the fuck am I writing here.
    The original list remains intact.

    :param attributes: the attributes to filter
    :param examples: the Examples by which to filter the attributes
    :return: a new list containing only the attributes that are present in at
    least one Example
    """

    def predicate(attribute):
        return any(attribute in example.attributes for example in examples)

    return list(filter(predicate, attributes))
