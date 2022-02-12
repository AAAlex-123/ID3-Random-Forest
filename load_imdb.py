import os

from file_util import reader
from id3 import Example, Category
from timed import Timer

Priority = Timer.Priority


@Timer(priority=Priority.RUN, prompt="Load Data")
def load_all_examples(directory: str, sample_size=5000) -> set[Example]:
    """ TODO """

    neg_dir_path = os.path.join(directory, "neg")
    pos_dir_path = os.path.join(directory, "pos")

    examples = set()
    examples |= load_examples_of_category(neg_dir_path, Category.POS, sample_size // 2)
    examples |= load_examples_of_category(pos_dir_path, Category.NEG, sample_size // 2)

    return examples


def load_examples_of_category(directory: str, category: Category, sample_size: int, verbose: bool = False) -> set[Example]:
    """ TODO """

    files = os.listdir(directory)
    examples = set()
    one_tenth_progress = min(len(files), sample_size) // 10

    for count, file in enumerate(files):
        if count == sample_size:
            break

        with reader(os.path.join(directory, file)) as f:
            contents = f.read().lower()

        examples.add(Example(category, contents))

        if verbose and (count + 1) % one_tenth_progress == 0:  # `count` is 0-indexed
            percent_done = (count + 1) // one_tenth_progress * 10
            print(f"\r{percent_done}% complete...", end="", flush=True)

    print()  # move cursor to the next line

    return examples


@Timer(priority=Priority.RUN, prompt="Load Attributes")
def load_all_attributes(filename: str, most_frequent: int, ignored: int) -> set[str]:
    # TODO should use a set of Examples to extract the attributes
    """ TODO """

    attributes = list()

    with reader(filename) as f:
        for _ in range(ignored):
            f.readline()

        for _ in range(most_frequent):
            attribute = f.readline().strip('\n')
            attributes.append(Example.sanitize_attribute(attribute))

    return set(attributes)


@Timer(priority=Priority.TEST)
def filter_attributes_by_examples(attributes: set[str], examples: set[Example]) -> set[str]:
    """ TODO """

    def predicate(attribute):
        return any(attribute in example.attributes for example in examples)

    return set(filter(predicate, attributes))
