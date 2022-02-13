
class io:
    def __init__(self, filename: str, mode: str):
        self.filename = filename
        self.mode = mode
        self._file_handle = None

    def __enter__(self):
        self.file_handle = open(self.filename, mode=self.mode, encoding="utf-8")
        return self.file_handle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_handle.close()


class reader(io):
    def __init__(self, filename: str):
        super().__init__(filename, 'r')


class writer(io):
    def __init__(self, filename: str):
        super().__init__(filename, 'w')

