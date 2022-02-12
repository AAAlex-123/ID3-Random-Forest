import time
from enum import IntEnum


class Timer:
    """ TODO """

    class Priority(IntEnum):
        """ TODO """

        LOW = 0
        MEDIUM = 3
        HIGH = 7
        CRITICAL = 10

        TEST = -1
        DEBUG = 0
        RUN = 5
        PRODUCTION = 10

    class Predicate:
        """ TODO """

        @staticmethod
        def and_(p1: 'Predicate', p2: 'Predicate'):
            """ TODO """
            return lambda x: p1(x) and p2(x)

        @staticmethod
        def or_(p1: 'Predicate', p2: 'Predicate'):
            """ TODO """
            return lambda x: p1(x) or p2(x)

        @staticmethod
        def not_(p1: 'Predicate'):
            """ TODO """
            return lambda x: not p1(x)

        @staticmethod
        def xor_(p1: 'Predicate', p2: 'Predicate'):
            """ TODO """
            return lambda x: p1(x) ^ p2(x)

        @staticmethod
        def above(lower_bound: int):
            """ TODO """
            return lambda x: x >= lower_bound

        @staticmethod
        def below(upper_bound: int):
            """ TODO """
            return lambda x: x <= upper_bound

        @staticmethod
        def equal(target: int):
            """ TODO """
            return lambda x: x == target

        @staticmethod
        def not_equal(target: int):
            """ TODO """
            return lambda x: x != target

        @staticmethod
        def between(lower_bound: int, higher_bound: int):
            """ TODO """
            return lambda x: lower_bound <= x <= higher_bound

        @staticmethod
        def true():
            """ TODO """
            return lambda x: True

        @staticmethod
        def false():
            """ TODO """
            return lambda x: False

    _enabled: bool = True
    _predicate = Predicate.true()

    def __init__(self, priority: int, prompt: str = ""):
        """ TODO """
        self._priority: int = priority
        self._prompt: str = prompt

    def __call__(self, func):

        prompt = self._prompt

        only_finish = prompt is None
        if prompt == "":
            prompt = func.__name__

        def wrapper(*args, **kwargs):
            can_time = Timer._enabled and Timer._predicate(self._priority)

            if can_time:
                start = time.perf_counter()
                if not only_finish:
                    print(f"Starting procedure: {prompt}")

            return_value = func(*args, **kwargs)

            if can_time:
                end = time.perf_counter()
                if not only_finish:
                    print(f"Procedure `{prompt}` finished after {end - start} seconds")
                else:
                    print(f"Finished after {end - start} seconds")

            return return_value

        return wrapper

    @classmethod
    def enable(cls) -> None:
        """ TODO """
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """ TODO """
        cls._enabled = False

    @classmethod
    def set_predicate(cls, predicate) -> None:
        """ TODO """
        cls._predicate = predicate
