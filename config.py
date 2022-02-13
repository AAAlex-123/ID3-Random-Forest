import json
import os

from file_util import reader, writer


class Config:
    """ TODO """

    def __init__(self, filename: str) -> None:
        """ TODO """

        self._filename = filename
        with reader(self._filename) as f:
            self._data = json.load(f)

        invalids = {item for item in self._data if not self.is_valid(item)}
        if len(invalids) > 0:
            msg = "The following key-value pairs are not valid:\n"
            raise ValueError(
                msg + '\n'.join(f"'{k}' - '{v}'" for k, v in invalids))

    def is_valid(self, item):
        """ TODO """

        def check_directory(path: str) -> bool:
            return os.path.isabs(path) and os.path.isdir(path)

        def check_file(path: str) -> bool:
            return os.path.isabs(path) and os.path.isfile(path)

        def check_num(num: int, low: int, high: int) -> bool:
            try:
                number = int(num)
            except ValueError:
                return False
            return low <= number <= high

        if item[0] == "directory":
            return check_directory(item[1])
        elif item[0] == "train" or item[0] == "test":
            return check_directory(os.path.join(self["directory"], item[1]))
        elif item[0] == "vocab":
            return check_file(os.path.join(self["directory"], item[1]))
        elif item[0] == "example_size":
            return check_num(item[1], 0, 25000)
        elif item[0] == "ignored_attributes":
            return check_num(item[1], 0, 89526)
        elif item[0] == "attribute_count":
            return check_num(item[1], 0, 89526 - self["ignored_attributes"])

        return False

    def __getitem__(self, item):
        """ TODO """

        return self._data.__getitem__(item)

    def __setitem__(self, key, value):
        """ TODO """

        if not Config.is_valid((key, value)):
            raise ValueError(f"Invalid key-value pair: '{key}' - '{value}'")

        self._data.__setitem__(key, value)

    def __enter__(self):
        return self._data

    def __exit__(self, exc_type, exc_val, exc_tb):
        with writer(self._filename) as f:
            json.dump(self._data, f, indent='\t')
