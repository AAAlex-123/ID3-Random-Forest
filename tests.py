import os

from timed import timed
from load_imdb import load_examples, load_attributes
from id3 import entropy


@timed(prompt="Test Load")
def test_load(sample_size: int, count: int, ignore: int) -> None:

    # path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_imdb = "C:\\Users\\alexm\\projects\\C++\\AI-algorithms\\resources\\aclImdb"
    example_dir = os.path.join(path_to_imdb, "train")
    attribute_dir = os.path.join(path_to_imdb, "imdb.vocab")

    try:
        examples = load_examples(example_dir, sample_size)
        attributes = load_attributes(attribute_dir, count, ignore)

    except os.error as err:
        print(f"Loading didn't complete normally due to: {err}")
        return

    for example in examples:
        print(example)

    print(attributes, sep="\n")


def test_entropy(*probabilities: float) -> list[float]:
    """
    Prints and returns the entropy for every probability provided and runs some hard-coded assertions

    :param probabilities: the probabilities to calculate the entropy for
    :return: a list that has the entropy of probability i at index i
    """
    entropies = [entropy(p) for p in probabilities]
    for probability, probability_entropy in zip(probabilities, entropies):
        print("H(%0.2f) = %0.5f" % (probability, probability_entropy))

    assert entropy(1) == 0
    assert entropy(0) == 0
    assert entropy(1/2) == 1

    return entropies

def find_best_cutoff():
    max_accuracy = -1
    max_cutoff = -1

    examples = 20000
    ignored_attrs = 25
    count_attrs = 200
    data_dir = sys.argv[1]

    example_list: list[Example] = list(load_examples(os.path.join(data_dir, "train"), examples))
    test_examples: set[Example] = set(example_list[:len(example_list)//2])
    train_examples: set[Example] = set(example_list[len(example_list)//2:])

    attributes: set[str] = load_attributes(os.path.join(sys.argv[1], "imdb.vocab"), count_attrs, ignored_attrs)

    for i in range(70, 100, 1):
        ID3.cutoff = i/100
        id3 = ID3(train_examples, attributes)
        results = test_classifier(id3, test_examples)

        if results.accuracy() > max_accuracy:
            max_accuracy = results.accuracy()
            max_cutoff = i/100

    return max_cutoff, max_accuracy


def find_best_tree_count():
    max_accuracy = -1
    max_cutoff = -1

    examples = 10000
    ignored_attrs = 25
    count_attrs = 200
    data_dir = sys.argv[1]

    example_list: list[Example] = list(load_examples(os.path.join(data_dir, "train"), examples))
    test_examples: set[Example] = set(example_list[:len(example_list)//2])
    train_examples: set[Example] = set(example_list[len(example_list)//2:])

    attributes: set[str] = load_attributes(os.path.join(sys.argv[1], "imdb.vocab"), count_attrs, ignored_attrs)

    for i in range(70, 201, 5):
        RandomForest.tree_count = i
        rand_forest = RandomForest(train_examples, attributes)
        results = test_classifier(rand_forest, test_examples)

        if results.accuracy() > max_accuracy:
            max_accuracy = results.accuracy()
            max_cutoff = i

        print("Trying", i, "trees, accuracy=", results.accuracy())

    return max_cutoff, max_accuracy


if __name__ == '__main__':
    print(find_best_tree_count())
