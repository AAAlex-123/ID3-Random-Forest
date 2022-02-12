from load_imdb import load_all_examples, load_all_attributes
from id3 import ID3
from random_forest import RandomForest
from test_stats import TestStats
from classifier import Classifier, Category, Example
from timed import timed, timed2

import os
import sys


class TestResults:
    def __init__(self, id3_train_results: TestStats, id3_test_results: TestStats, forest_train_results: TestStats,
                 forest_test_results: TestStats):
        self.id3_train_results = id3_train_results
        self.id3_test_results = id3_test_results
        self.forest_train_results = forest_train_results
        self.forest_test_results = forest_test_results


def test_classifier(classifier: Classifier, examples: set[Example]) -> TestStats:
    """
    Run a test on a given classifier for a given dataset.
    :param classifier: any Classifier object
    :param examples: the dataset to be tested
    :return: a TestStats object describing the results of the testing
    """
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    classifier.classify_bulk(examples)

    for example in examples:
        if example.actual == Category.POS and example.predicted == Category.POS:
            true_positives += 1
        elif example.actual == Category.NEG and example.predicted == Category.NEG:
            true_negatives += 1
        elif example.actual == Category.POS and example.predicted == Category.NEG:
            false_negatives += 1
        elif example.actual == Category.NEG and example.predicted == Category.POS:
            false_positives += 1

    return TestStats(true_negatives, true_positives, false_positives, false_negatives)


# @timed2()
def main_test(train_data_dir: str, test_data_dir: str, vocab_file_dir: str,
              example_size: int, attr_count: int, ignore_attr_count: int) -> TestResults:
    train_data = load_all_examples(train_data_dir, example_size)
    testing_data = load_all_examples(test_data_dir, example_size)
    attributes = load_all_attributes(vocab_file_dir, attr_count, ignore_attr_count)

    id3 = ID3.create_timed(train_data, attributes)
    rand_forest = RandomForest.create_timed(train_data, attributes)

    return TestResults(test_classifier(id3, train_data), test_classifier(id3, testing_data),
                       test_classifier(rand_forest, train_data), test_classifier(rand_forest, testing_data))


@timed(prompt="Main program")
def main() -> None:
    def check_int_arg(arg: str, param_name: str, bottom_limit: int, upper_limit: int) -> int:
        try:
            number = int(arg)
        except ValueError:
            number = -1

        if number < bottom_limit or number > upper_limit:
            print(f"Error: parameter `{param_name}` must be a valid integer within the "
                  f"[{bottom_limit}, {upper_limit}] range")
            sys.exit(1)
        else:
            return number

    if len(sys.argv) < 5:
        print("Insufficient parameters:")
        print("Use main.py <imdb directory> <number of examples to be loaded>"
              " <number of ignored words> <number of words to be considered>")
    else:
        data_dir = sys.argv[1]
        train_data_dir = os.path.join(data_dir, "train")
        test_data_dir = os.path.join(data_dir, "test")
        vocab_file_name = os.path.join(data_dir, "imdb.vocab")

        example_size = check_int_arg(sys.argv[2], "example size", 100, 250000)
        ignore_attr_count = check_int_arg(sys.argv[3], "ignored words count", 0, 90000)
        attr_count = check_int_arg(sys.argv[4], "total word count", 5, 90000)

        if attr_count > 200:
            answer = input("Warning: Giving more than 200 words as attributes is likely to make the algorithm "
                           "incredibly inconsistent. Are you sure you want to proceed? (y/n)")
            if answer.lower() != "y":
                sys.exit(0)

        results = main_test(train_data_dir, test_data_dir, vocab_file_name, example_size, attr_count, ignore_attr_count)
        print("ID3 training results: ", results.id3_train_results)
        print("ID3 testing results: ", results.id3_test_results)
        print("Random Forest training results: ", results.forest_train_results)
        print("Random Forest testing results: ", results.forest_test_results)


if __name__ == "__main__":
    try:
        main()
    except IOError as ioe:
        print("Error: ", ioe)
        print("Please make sure the first argument is the upper-most directory ('aclImdb') containing the imdb dataset")
