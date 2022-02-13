import os
import sys

from file_util import writer
from main import get_results_from_all_classifications
from timed import Timer

Priority = Timer.Priority


@Timer(priority=0, prompt="Get Plotting Data")
def get_plotting_data(data_dir: str) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    Get the accuracy of repeated executions of the ID3 and Random Forest classifiers with
    variable parameters.

    :param data_dir: the directory containing the training examples
    :return: A tuple containing 4 lists of floats. More specifically:

    0: A list with the accuracy of the ID3 tree using the training data
    1: A list with the accuracy of the ID3 tree using the testing data
    2: A list with the accuracy of Random Forest using the training data
    3: A list with the accuracy of Random Forest using the testing data
    """

    train_data_dir = os.path.join(data_dir, "train")
    test_data_dir = os.path.join(data_dir, "test")
    vocab_file_name = os.path.join(data_dir, "imdb.vocab")

    results = dict()
    results['counts'] = []
    results['id3'] = {'tr': [], 'te': []}
    results['rf'] = {'tr': [], 'te': []}

    ignored_attrs = 12
    count_attrs = 200
    
    print("ID3 - R/F - TRA - TES - Count - {:^10s} - {:^10s} - {:^10s} - {:^10s} ".format("Accuracy", "Precision", "Recall", "F1"))

    for example_count in [250, 500] + list(range(1000, 10001, 1000)):
        result = get_results_from_all_classifications(train_data_dir, test_data_dir, vocab_file_name, example_count, count_attrs,
                                                      ignored_attrs)
        id3tr = result.id3_tr
        id3te = result.id3_te
        frtr = result.rf
        frte = result.rf
        
        r = id3tr
        print(" X  -     -  X  -     - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = id3te
        print(" X  -     -     -  X  - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = frtr
        print("    -  X  -  X  -     - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = frte
        print("    -  X  -     -  X  - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        
        results['counts'].append(example_count)
        results['id3']['tr'].append(id3tr)
        results['id3']['te'].append(id3te)
        results['rf']['tr'].append(frtr)
        results['rf']['te'].append(frte)

    return results


@Timer(priority=Priority.PRODUCTION, prompt="Write data to csv")
def write_data_to_files(data, output_file_name: str, methods: list[str], args: dict[str, list]) -> None:
    """
    TODO

    :param data: the directory containing the training examples
    :param output_file_name: The name the output will be saved to
    :param methods:
    :param args:
    """

    header = "count," + ",".join(m for m in methods)

    for algorithm in ["id3", "rf"]:
        for mode in ["tr", "te"]:
            with writer(f"{output_file_name}_{algorithm}_{mode}.csv") as f:
                f.write(f"{header}\n")
                for count in data["count"]:
                    f.write(f"{count}")
                    classifier_evaluation = data[algorithm][mode][count]
                    for method_name in methods:
                        method = getattr(classifier_evaluation, method_name)
                        arguments = (args.get(method_name))
                        f.write(f",{method(*arguments)}")

                # f.write("\n".join([header] + [f"{count}," + ",".join(str(getattr(data[algorithm][mode][count], method)(*args.get(method))) for method in methods)]) for count in data["count"])


@Timer(priority=Priority.PRODUCTION, prompt="Get Data for Graphs")
def main():
    try:
        data = get_plotting_data(sys.argv[1])
        write_data_to_files(data, sys.argv[2])
    except IOError as ioe:
        print("Error: ", ioe)


if __name__ == "__main__":
    main()
