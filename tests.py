import os

from classifier import Category, Example
from classifier_evaluation import ClassifierEvaluation
from timed import Timer
from load_imdb import load_examples_from_directory, load_attributes_from_file
from id3_util import _entropy

TEST = Timer.Priority.TEST


@Timer(priority=-1, prompt="Test Load")
def test_load(sample_size: int, count: int, ignore: int) -> None:

    # path_to_file = os.path.dirname(os.path.abspath(__file__))
    path_to_imdb = "C:\\Users\\alexm\\projects\\C++\\AI-algorithms\\resources\\aclImdb"
    example_dir = os.path.join(path_to_imdb, "train")
    attribute_dir = os.path.join(path_to_imdb, "imdb.vocab")

    try:
        examples = load_examples_from_directory(example_dir, sample_size)
        attributes = load_attributes_from_file(attribute_dir, count, ignore)

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
    entropies = [_entropy(p) for p in probabilities]
    for probability, probability_entropy in zip(probabilities, entropies):
        print("H(%0.2f) = %0.5f" % (probability, probability_entropy))

    assert _entropy(1) == 0
    assert _entropy(0) == 0
    assert _entropy(1 / 2) == 1

    return entropies


@Timer(priority=TEST)
def test_classifier_evaluation() -> ClassifierEvaluation:
    p = Category.POS
    n = Category.NEG
    act = [p, p, p, p, p, n, n, n, n, n]
    pre = [p, p, p, p, n, p, p, p, n, n]
    examples = [Example(Category.NONE, "") for _ in range(10)]
    for ac, pr, ex in zip(act, pre, examples):
        ex.actual = ac
        ex.predicted = pr

    ce = ClassifierEvaluation(set(examples))
    assert ce.accuracy() == 0.6
    assert ce.precision(p) == 0.5714285714285714
    assert ce.recall(p) == 0.8
    assert ce.f_measure(p, 1) == 0.6666666666666666

    return ce


def find_best_cutoff():
    max_accuracy = -1
    max_cutoff = -1

    examples = 20000
    ignored_attrs = 25
    count_attrs = 200
    data_dir = sys.argv[1]

    example_list: list[Example] = list(load_examples_from_directory(os.path.join(data_dir, "train"), examples))
    test_examples: set[Example] = set(example_list[:len(example_list)//2])
    train_examples: set[Example] = set(example_list[len(example_list)//2:])

    attributes: set[str] = load_attributes_from_file(os.path.join(sys.argv[1], "imdb.vocab"), count_attrs, ignored_attrs)

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

    example_list: list[Example] = list(load_examples_from_directory(os.path.join(data_dir, "train"), examples))
    test_examples: set[Example] = set(example_list[:len(example_list)//2])
    train_examples: set[Example] = set(example_list[len(example_list)//2:])

    attributes: set[str] = load_attributes_from_file(os.path.join(sys.argv[1], "imdb.vocab"), count_attrs, ignored_attrs)

    for i in range(70, 201, 5):
        RandomForest.tree_count = i
        rand_forest = RandomForest(train_examples, attributes)
        results = test_classifier(rand_forest, test_examples)

        if results.accuracy() > max_accuracy:
            max_accuracy = results.accuracy()
            max_cutoff = i

        print("Trying", i, "trees, accuracy=", results.accuracy())

    return max_cutoff, max_accuracy


def test_timer():
    import time

    @Timer(priority=Timer.Priority.LOW)
    def add1(a: int, b: int, n: float) -> int:
        time.sleep(n)
        return a * b

    @Timer(priority=Timer.Priority.MEDIUM, prompt="my prompt")
    def add2(a: int, b: int, n: float) -> int:
        time.sleep(n)
        return a * b

    @Timer(priority=Timer.Priority.HIGH, prompt=None)
    def add3(a: int, b: int, n: float) -> int:
        time.sleep(n)
        return a * b

    Timer.set_predicate(Timer.Predicate.above(6))
    print("set to above 6")
    add1(1, 2, 0.5)
    add2(2, 3, 0.5)
    add3(3, 4, 0.5)
    Timer.set_predicate(Timer.Predicate.equal(7))
    print("set to equal 7")
    add1(1, 2, 0.5)
    add2(2, 3, 0.5)
    add3(3, 4, 0.5)
    Timer.set_predicate(Timer.Predicate.between(3, 8))
    print("set to between 3, 8")
    add1(1, 2, 0.5)
    add2(2, 3, 0.5)
    add3(3, 4, 0.5)


if __name__ == '__main__':
    Timer.set_predicate(Timer.Predicate.equal(TEST))
    test_classifier_evaluation()
