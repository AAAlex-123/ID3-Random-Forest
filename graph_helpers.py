import os
import sys
from csv import excel

from main import main_test
from timed import timed


@timed(prompt="Get Overfit Data")
def get_overfit_data(data_dir: str) -> tuple[list[float], list[float], list[float], list[float]]:
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

    counts = []
    id3_train_results = []
    id3_test_results = []
    forest_train_results = []
    forest_test_results = []

    ignored_attrs = 12
    count_attrs = 200
    
    print("ID3 - R/F - TRA - TES - Count - {:^10s} - {:^10s} - {:^10s} - {:^10s} ".format("Accuracy", "Precision", "Recall", "F1"))

    # get overfitting curve data
    for example_count in [250, 500] + list(range(1000, 10001, 1000)):
        # print("OVERFIT FOR examples: %5d" % example_count, end="\t")
        result = main_test(train_data_dir, test_data_dir, vocab_file_name, example_count, count_attrs,
                           ignored_attrs)
        id3tr = result.id3_train_results
        id3te = result.id3_test_results
        foresttr = result.forest_train_results
        forestte = result.forest_test_results
        
        r = id3tr
        print(" X  -     -  X  -     - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = id3te
        print(" X  -     -     -  X  - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = foresttr
        print("    -  X  -  X  -     - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        r = forestte
        print("    -  X  -     -  X  - ", end="")
        print("%5d - %.8f - %.8f - %.8f - %.8f" % (example_count, r.accuracy(), r.precision(), r.recall(), r.f_measure()))
        
        counts.append(example_count)
        id3_train_results.append(id3tr)
        id3_test_results.append(id3te)
        forest_train_results.append(foresttr)
        forest_test_results.append(forestte)

    return counts, id3_train_results, id3_test_results, forest_train_results, forest_test_results


@timed(prompt="Get ROC Data")
def get_roc_data(data_dir: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]],
                                         list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Get the precision and recall of repeated executions of the ID3 and Random Forest classifiers with
    variable parameters.
    :param data_dir: the directory containing the training examples
    :return: A tuple containing 4 lists tuples holding 2 floats each. More specifically:
    0: A list with the precision and recall of the ID3 tree using the training data
    1: A list with the precision and recall of the ID3 tree using the testing data
    2: A list with the precision and recall of Random Forest using the training data
    3: A list with the precision and recall of Random Forest using the testing data
    """
    train_data_dir = os.path.join(data_dir, "train")
    test_data_dir = os.path.join(data_dir, "test")
    vocab_file_name = os.path.join(data_dir, "imdb.vocab")

    id3_train_results: list[tuple[float, float]] = []
    id3_test_results: list[tuple[float, float]] = []
    forest_train_results: list[tuple[float, float]] = []
    forest_test_results: list[tuple[float, float]] = []

    # get ROC curve data
    for ignored_attrs in range(0, 201, 100):
        for count_attrs in range(50, 251, 100):
            for example_count in [500, 1000] + list(range(2000, 10001, 2000)):
                print("ROC FOR ignored-attrs-examples: %3d %3d %5d" % (ignored_attrs, count_attrs, example_count), end="\t\t")
                results = main_test(train_data_dir, test_data_dir, vocab_file_name, example_count, count_attrs,
                                    ignored_attrs)
                id3_train_results.append((results.id3_train_results.precision(), results.id3_train_results.recall()))
                id3_test_results.append((results.id3_test_results.precision(), results.id3_test_results.recall()))
                forest_train_results.append((results.forest_train_results.precision(),
                                             results.forest_train_results.recall()))
                forest_test_results.append((results.forest_test_results.precision(),
                                            results.forest_test_results.recall()))

    return id3_train_results, id3_test_results, forest_train_results, forest_test_results


@timed(prompt="Overfit To File")
def overfit_to_file(data_dir: str, output_file_name: str) -> None:
    """
    Write the formatted overfit data into an output file.
    :param data_dir: the directory containing the training examples
    :param output_file_name: The name the output will be saved to
    """
    data = get_overfit_data(data_dir)
    
    for i, fname in enumerate(["id3", "rf"]):
        for j, mode in enumerate(["tr", "te"]):
            with open(f"{output_file_name}_{fname}_accuracy_{mode}.csv", mode="w") as f:
                f.write(f"count,accuracy_{mode}\n")
                for count, stat in zip(data[0], data[2*i+1+j]):
                    f.write(f"{count},{stat.accuracy()}\n")

            with open(f"{output_file_name}_{fname}_prec_rec_f_{mode}.csv", mode="w") as f:
                f.write(f"count,accuracy_{mode},recall_{mode},f_measure_{mode}\n")
                for count, stat in zip(data[0], data[2*i+1+j]):
                    f.write(f"{count},{stat.precision()},{stat.recall()},{stat.f_measure()}\n")
        
    
    
    """
    separators = ["Example Count:\n", "Training id3 accuracy:\n", "Testing id3 accuracy:\n", "Testing random forest accuracy:\n",
                  "Testing random forest accuracy:\n"]
    out_str = ""
    for i, separator in enumerate(separators):
        out_str += separator
        if i == 0:
            out_str += ", ".join([str(count) for count in data[i]])
        else:
            out_str += ", ".join([str(test_stat.accuracy()) for test_stat in data[i]])
        out_str += "\n"

    with open(output_file_name, mode="w") as out_of:
        out_of.write(out_str)
    """


@timed(prompt="ROC To File")
def roc_to_file(data_dir: str, output_file_name: str):
    """
    Write the formatted precision / recall data into an output file.
    :param data_dir: the directory containing the training examples
    :param output_file_name: The name the output will be saved to
    """
    data = get_roc_data(data_dir)
    separators = ["Example Count:\n", "Training id3 precision/recall:\n", "Testing id3 precision/recall:\n",
                  "Testing random forest precision/recall:\n", "Testing random forest precision/recall:\n"]
    out_str = ""
    for i, separator in enumerate(separators):
        out_str += separator
        if i == 0:
            out_str += ", ".join([str(count) for count in data[i]])
        else:
            out_str += ", ".join([str(test_stat.accuracy()) for test_stat in data[i]])
        out_str += "\n"
        """
        out_str += separator
        for precision, recall in data[i]:
            out_str += f"({str(precision)}, {str(recall)}), "
        out_str = out_str[:-1]  # remove last comma
        out_str += "\n"
        """

    with open(output_file_name, mode="w") as out_of:
        out_of.write(out_str)


@timed(prompt="Get Data for Graphs")
def main():
    try:
        overfit_to_file(sys.argv[1], sys.argv[2])
        # roc_to_file(sys.argv[1], sys.argv[2])
    except IOError as ioe:
        print("Error: ", ioe)


if __name__ == "__main__":
    main()
