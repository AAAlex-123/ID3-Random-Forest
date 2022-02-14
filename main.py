import sys

from classifier_evaluation import ClassifierEvaluation
from config import Config
from load_imdb import load_examples_from_directory, load_attributes_from_file,\
    filter_attributes_by_examples
from classifier import Classifier, Example
from id3 import ID3
from random_forest import RandomForest
from timed import Timer

Priority = Timer.Priority
CE = ClassifierEvaluation


@Timer(priority=Priority.RUN)
def load_data(config: Config):
    """ TODO """

    examples_tr = load_examples_from_directory(config["train"], config["example_count"])
    examples_te = load_examples_from_directory(config["test"], config["example_count"])

    attributes = load_attributes_from_file(config["vocab"],
                                           config["attribute count"],
                                           config["ignored_attributes"])
    attributes = filter_attributes_by_examples(attributes, examples_tr)

    return examples_tr, examples_te, attributes


@Timer(priority=Priority.RUN)
def get_trained_classifiers(examples: set[Example], attributes: set[str]) ->\
        tuple[ID3, RandomForest]:
    """ TODO """

    # no need to copy the Examples, random forest makes a copy for each tree
    id3 = ID3(examples, attributes)
    random_forest = RandomForest(examples, attributes)

    return id3, random_forest


@Timer(priority=Priority.RUN)
def classify_and_stats(classifier: Classifier, examples: set[Example]) -> CE:
    """
    TODO

    :param classifier: a trained Classifier
    :param examples: the Examples that will be classified using the Classifier
    :return: a ClassifierEvaluation object with the results of the classification
    """

    classifier.classify_bulk([Example.copy_of(e) for e in examples])

    return CE(examples)


class Results:
    """ TODO """

    def __init__(self, id3_tr: CE, id3_te: CE, fr_tr: CE, fr_te: CE):
        self.id3_tr = id3_tr
        self.id3_te = id3_te
        self.rf = fr_tr
        self.rf = fr_te


@Timer(priority=Priority.RUN)
def get_results_from_all_classifications(config: Config) -> Results:
    """ TODO """

    examples_tr, examples_te, attributes = load_data(config)
    id3, rand_forest = get_trained_classifiers(examples_tr, attributes)

    return Results(classify_and_stats(id3, examples_tr),
                   classify_and_stats(id3, examples_te),
                   classify_and_stats(rand_forest, examples_tr),
                   classify_and_stats(rand_forest, examples_te))


@Timer(priority=Priority.PRODUCTION, prompt="Main program")
def main() -> None:
    """ TODO """

    if len(sys.argv) != 2:
        print("Usage:\npython main.py <config file>")
        return

    config = Config(sys.argv[1])

    results = get_results_from_all_classifications(config)

    print("ID3 training results: ", results.id3_tr.basic_stats())
    print("ID3 testing results: ", results.id3_te.basic_stats())
    print("Random Forest training results: ", results.rf.basic_stats())
    print("Random Forest testing results: ", results.rf.basic_stats())


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(f"An Exception occurred:\n\n{exception}", file=sys.stderr)
