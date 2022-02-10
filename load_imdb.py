import os
from id3 import Example, Category
from timed import timed
import re


# @timed(prompt="Load Data")
def load_examples(directory: str, sample_size=5000) -> set[Example]:
    neg_dir_path = os.path.join(directory, "neg")
    pos_dir_path = os.path.join(directory, "pos")

    examples = set()
    examples |= _load_examples(neg_dir_path, Category.POS, sample_size // 2)
    examples |= _load_examples(pos_dir_path, Category.NEG, sample_size // 2)

    return examples


def _load_examples(directory: str, category: Category, sample_size: int) -> set[Example]:
    # print(f"Loading {category.name} data")
    files = os.listdir(directory)
    sep = os.sep
    examples = set()
    one_tenth_progress = min(len(files), sample_size) // 10

    for count, file in enumerate(files):
        if count == sample_size:
            break

        with open(directory + sep + file, mode='r', encoding="utf8") as f_handle:
            contents = f_handle.read().lower()

        examples.add(Example(category, contents))

        if (count + 1) % one_tenth_progress == 0:  # `count` is 0-indexed
            pass
            # print('\b' * 20 + "%d%% complete..." % ((count + 1) // one_tenth_progress * 10), end="")

    # print()

    return examples

# @timed(prompt="Load Attributes")
def load_attributes(filename: str, most_freq: int, ignored: int) -> set[str]:

    attributes = list()

    with open(filename, mode='r', encoding="utf8") as f:
        for _ in range(ignored):
            f.readline()
        for _ in range(most_freq):
            attribute = f.readline().strip('\n')
            sanitized_attribute = Example._ignored_chars_pattern.sub("", attribute, 0)
            sanitized_attribute = re.sub("\\s+", " ", sanitized_attribute)
            attributes.append(sanitized_attribute)

    return set(attributes)
