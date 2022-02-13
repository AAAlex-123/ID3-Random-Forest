from __future__ import annotations

import sys

from config import Config
from file_util import writer
from main import get_results_from_all_classifications
from timed import Timer

Priority = Timer.Priority


@Timer(priority=0, prompt="Get Plotting Data")
def get_plotting_data(config: Config) -> dict[str, list | dict[str, list]]:
    """
    TODO

    Get the accuracy of repeated executions of the ID3 and Random Forest classifiers with
    variable parameters.

    :param config: TODO
    :return: A tuple containing 4 lists of floats. More specifically:

    0: A list with the accuracy of the ID3 tree using the training data
    1: A list with the accuracy of the ID3 tree using the testing data
    2: A list with the accuracy of Random Forest using the training data
    3: A list with the accuracy of Random Forest using the testing data
    """

    results: dict[str, list | dict[str, list]] = dict()
    results['counts']: list = []
    results['id3']: dict[str, list] = {'tr': [], 'te': []}
    results['rf']: dict[str, list] = {'tr': [], 'te': []}

    config['ignored_attributes'] = 12
    config['attribute_count'] = 200

    fmt = "ID3 - R/F - TRA - TES - Count" + 4 * " - {:^10s}"
    print(fmt.format("Accuracy", "Precision", "Recall", "F1"))

    for example_count in [250, 500] + list(range(1000, 10001, 1000)):
        config['example_count'] = example_count
        result = get_results_from_all_classifications(config)

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
def write_data_to_files(data, output_file_name: str, methods: list[str], args: dict[str, tuple]) -> None:
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
    """ TODO """

    if len(sys.argv) != 3:
        print("Usage:\npython graph_helpers.py <config file> <output file>")
        return

    config = Config(sys.argv[1])

    data = get_plotting_data(config)
    write_data_to_files(data, sys.argv[2])


if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        print(f"An Exception occurred:\n\n{exception}", file=sys.stderr)
