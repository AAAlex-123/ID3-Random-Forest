import json

from file_util import reader, writer


class Config:

    def __init__(self, filename: str):
        self._filename = filename
        with reader(self._filename) as f:
            self._data = json.load(f)

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def __setitem__(self, key, value):
        return self._data.__setitem__(key, value)

    def __enter__(self):
        return self._data

    def __exit__(self, exc_type, exc_val, exc_tb):
        with writer(self._filename) as f:
            json.dump(self._data, f, indent='\t')
