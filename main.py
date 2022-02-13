import os
import sys

from classifier_evaluation import ClassifierEvaluation
from load_imdb import load_all_examples, load_all_attributes, filter_attributes_by_examples
from classifier import Classifier, Example
from id3 import ID3
from random_forest import RandomForest
from timed import Timer

Priority = Timer.Priority
CE = ClassifierEvaluation


@Timer(priority=Priority.RUN)
def load_data(dir_tr: str, dir_te: str, vocab_file: str, example_count: int, attr_count: int, ignored_attr_count: int):
    """ TODO """

    examples_tr = load_all_examples(dir_tr, example_count)
    examples_te = load_all_examples(dir_te, example_count)
    attributes = load_all_attributes(vocab_file, attr_count, ignored_attr_count)
    attributes = filter_attributes_by_examples(attributes, examples_tr)

    return examples_tr, examples_te, attributes


@Timer(priority=Priority.RUN)
def get_trained_classifiers(examples, attributes) -> tuple[ID3, RandomForest]:
    """ TODO """

    # no need to copy the examples since random forest makes a copy for each tree
    id3 = ID3(examples, attributes)
    random_forest = RandomForest(examples, attributes)

    return id3, random_forest


@Timer(priority=Priority.RUN)
def get_stats_from_classifier(classifier: Classifier, examples: set[Example]) -> CE:
    """
    TODO

    :param classifier: a trained Classifier
    :param examples: the Examples that will be classified using the Classifier
    :return: a ClassifierEvaluation object with the results of the classification
    """

    classifier.classify_bulk(examples)

    return CE(examples)


class Results:
    """ TODO """

    def __init__(self, id3_tr: CE, id3_te: CE, fr_tr: CE, fr_te: CE):
        self.id3_tr = id3_tr
        self.id3_te = id3_te
        self.rf = fr_tr
        self.rf = fr_te


@Timer(priority=Priority.RUN)
def get_results_from_all_classifications(dir_tr: str, dir_te: str, vocab_file: str, example_count: int, attr_count: int, ignored_attr_count: int) -> Results:

    data = load_data(dir_tr, dir_te, vocab_file, example_count, attr_count, ignored_attr_count)
    examples_tr, examples_te, attributes = data

    id3, rand_forest = get_trained_classifiers(examples_tr, attributes)

    return Results(get_stats_from_classifier(id3, examples_tr), get_stats_from_classifier(id3, examples_te),
                   get_stats_from_classifier(rand_forest, examples_tr), get_stats_from_classifier(rand_forest, examples_te))


@Timer(priority=Priority.PRODUCTION, prompt="Main program")
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

        results = get_results_from_all_classifications(train_data_dir, test_data_dir, vocab_file_name, example_size, attr_count, ignore_attr_count)
        print("ID3 training results: ", results.id3_tr.basic_stats())
        print("ID3 testing results: ", results.id3_te.basic_stats())
        print("Random Forest training results: ", results.rf.basic_stats())
        print("Random Forest testing results: ", results.rf.basic_stats())


if __name__ == "__main__":
    try:
        main()
    except IOError as ioe:
        print("Error: ", ioe)
        print("Please make sure the first argument is the upper-most directory ('aclImdb') containing the imdb dataset")
