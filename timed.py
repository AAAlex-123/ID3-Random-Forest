"""
Defines the Timer decorator and other utility classes.

The Timer decorates a function, times its execution time and logs information
about it each time it is executed. It is highly configurable using the
following classes in this module:

Priority: Defines some enum constants that can be used to provide consistent
usage of the priority field of a Timer. That field is used by a Predicate to
determine if a Timer should run.

Predicate: Simplifies the creation of complex predicate functions that
determine whether a Timer with a specific Priority should run at a specific
moment at runtime.

TimeLogger hierarchy: The abstract TimeLogger class outlines the process of
logging timing information about a block of code. It is used as a context
manager in a with-statement. The concrete classes that extend it, need only
define what and how exactly is logged at the beginning and end of execution by
overriding two abstract methods.
"""

from abc import ABC, abstractmethod

import time
from enum import IntEnum
from functools import reduce


class Priority(IntEnum):
    """ Defines a set of constants that may be used when decorating with @Timer

    The Priority associated with a Timer decorator determines whether the
    decorated function will actually be timed or not. The Priority is merely an
    integer that is compared against the Timer's global Predicate to determine
    if it will be applied or not. The constants defined here may be used for a
    more uniformed Priority system, as opposed to simple integers.
    """

    LOW = 0
    MEDIUM = 3
    HIGH = 7
    CRITICAL = 10

    TEST = -1
    DEBUG = 0
    RUN = 5
    PRODUCTION = 10


class Predicate:
    """ Defines a collection of static methods to compose Predicates

     A Predicate is a function that takes an integer, a Priority, and returns
     True or False. It is used with a Timer to control which of the decorated
     functions will actually be timed.

     The static methods of this class return lambda functions that do precisely
     this. Additionally, methods for locally and-ing, or-ing, xor-ing and
     negating other Predicates are defined. This enables Predicates to be
     composed with arbitrary complexity. Two special Predicates that always
     return True and False respectively are also defined.

     For example, Predicates can be composed like this:

     not(or(between(1, 4), equal(7)))

     to return True for any number that is outside the range 1, 4 and is also
     not equal to 7. This can of course be rewritten as:

     and(not(between(1, 4)), not(equal(7))
     """

    @staticmethod
    def and_(*predicate):
        """ Returns True if all predicates also return True """
        return lambda x: reduce(lambda p1, p2: p1(x) and p2(x), *predicate)

    @staticmethod
    def or_(*predicate):
        """ Returns True if any predicate also return True """
        return lambda x: reduce(lambda p1, p2: p1(x) or p2(x), *predicate)

    @staticmethod
    def not_(p1):
        """ Returns True iff p1 returns False """
        return lambda x: not p1(x)

    @staticmethod
    def xor_(*predicate):
        """ Returns True if an odd number of predicates also return True """
        return lambda x: reduce(lambda p1, p2: p1(x) ^ p2(x), *predicate)

    @staticmethod
    def above(lower_bound: int):
        """ Returns a predicate for the '&gt=' operator.

        :param lower_bound: the lower bound to check against
        :return: lambda x: x &gt= lower_bound
        """
        return lambda x: x >= lower_bound

    @staticmethod
    def below(upper_bound: int):
        """ Returns a predicate for the '&lt=' operator.

        :param upper_bound: the upper bound to check against
        :return: lambda x: x &lt= upper_bound
        """
        return lambda x: x <= upper_bound

    @staticmethod
    def equal(target: int):
        """ Returns a predicate for the '==' operator.

        :param target: the target number to check against
        :return: lambda x: x == target
        """
        return lambda x: x == target

    @staticmethod
    def not_equal(target: int):
        """ Returns a predicate for the '!=' operator.

        :param target: the target number to check against
        :return: lambda x: x != target
        """
        return lambda x: x != target

    @staticmethod
    def between(lower_bound: int, upper_bound: int):
        """ Returns a predicate for the 'a &lt= x &lt b' operator.

        :param lower_bound: the lower bound to check against
        :param upper_bound: the upper bound to check against
        :return: lambda x: lower_bound &lt= x &lt= higher_bound
        """
        return lambda x: lower_bound <= x <= upper_bound

    @staticmethod
    def true():
        """ Returns a predicate that always returns True.

        :return: lambda x: True
        """
        return lambda x: True

    @staticmethod
    def false():
        """ Returns a predicate that always returns False.

        :return: lambda x: False
        """
        return lambda x: False


class TimeLogger(ABC):
    """ Defines the abstract functionality of a Time Logger

    A Time Logger times the execution time of some code and additionally logs
    it in some fashion which can be customised by subclassing this class.

    To use this class enclose the code whose time will be logged in a 'with'
    statement and call an instance of the Logger by providing a prompt that
    will be used when logging as follows:

    with logger_instance("calculate ln2"):
        ln2 = sum((-1)**(i+1) / i for i in range(1, 10000000))

    To subclass this class, define how time will be logged at the start and end
    of execution, by implementing the abstract _log_start and _log_end methods.

    Note: due to the additional overhead that comes from the use of __enter__
    and __exit__ methods, the timing will be slightly inaccurate.
    """

    def __init__(self):
        self._prompt = ""
        self._start = 0
        self._end = 0

    def __call__(self, prompt: str):
        self._prompt = prompt
        return self

    def __enter__(self):
        self._log_start()
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()
        self._log_end(self._end - self._start)

    @abstractmethod
    def _log_start(self) -> None:
        pass

    @abstractmethod
    def _log_end(self, elapsed: float) -> None:
        pass


class DefaultLogger(TimeLogger):
    """ A TimeLogger that prints a both a start and end message

    The start message indicates that the procedure with the given prompt has
    started and the end message additionally displays the elapsed time in
    seconds.
    """

    def __init__(self):
        super(DefaultLogger, self).__init__()

    def _log_start(self):
        print(f"Starting procedure: {self._prompt}")

    def _log_end(self, elapsed: float):
        print(f"Procedure `{self._prompt}` finished after {elapsed} seconds")


class EndOnlyLogger(TimeLogger):
    """ A TimeLogger that only prints the end message

    The end message displays just the time elapsed in seconds.
    """

    def __init__(self):
        super(EndOnlyLogger, self).__init__()

    def _log_start(self):
        pass

    def _log_end(self, elapsed: float):
        print(f"Finished after {elapsed} seconds")


# TODO figure out how to make logging to files happen properly

"""
Logger will most likely be a decorator like:
StdOut/ErrLogger|FileLogger (*a, **kw, Default/EndOnlyLogger(*a, *kw))

class FileLogger(_TimeLogger):
    def __init__(self, filename: str):
        super(FileLogger, self).__init__()
        self._filename = filename

    def _log_start(self):
        with open(self._filename, mode="a", encoding="utf-8") as f:
            f.write(f"Starting procedure: {self._prompt}")

    def _log_end(self, elapsed: float):
        pass
"""


class Timer:
    """ A decorator that times and logs the execution time of a function

    For each function that it decorates, it keeps track of the execution time
    and additionally logs a message at the start and end of the function call.

    The functionality of the Timer is highly customisable. Using its class
    methods it can be enabled or disabled globally as well as configured with a
    Predicate to only time certain functions depending on the current needs of
    the application. These parameters can be changed at any time at run-time.

    Each individual function that is timed can have a different priority,
    depending on when it is desired to be timed, a different prompt, which is
    displayed when the function is logged, and a different TimeLogger instance,
    which alters the way that the execution information is logged. These
    parameters are specific to the function being timed and are defined when
    it is being decorated.

    Note: always use parentheses when decorating, as in @Timer(), even if no
    parameters are actually specified. Failing to do so will raise an
    AttributeError when the decorated function is called at runtime.
    """

    _enabled = True
    _predicate = Predicate.true()

    def __init__(self, priority: int = 0, prompt: str = "",
                 logger: TimeLogger = DefaultLogger()):
        """ Constructs a Timer

        :param priority: the priority that will be checked against the Timer's
        global Predicate to determine if the function will be timed each time
        it is executed.
        :param prompt: the prompt that may be displayed in the log message. If
        omitted or "", the name of the function being decorated is used.
        :param logger: the logger instance that will be used to log the timing
        information.
        """

        self._priority = priority
        self._prompt = prompt
        self._logger = logger

    def __call__(self, func):

        prompt = self._prompt
        if prompt == "":
            prompt = func.__name__

        def wrapper(*args, **kwargs):
            """
            Wraps the decorated function to time and log its execution time

            :param args: the args to be passed to the decorated function
            :param kwargs: the kwargs to be passed to the decorated function
            :return: whatever the decorated function returned
            """

            can_time = Timer._enabled and Timer._predicate(self._priority)

            if can_time and self._logger is not None:
                with self._logger(prompt):
                    return_value = func(*args, **kwargs)
            else:
                return_value = func(*args, **kwargs)
            return return_value

        return wrapper

    @classmethod
    def enable(cls) -> None:
        """ Globally enables the Timer functionality """
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """ Globally disables the Timer functionality """
        cls._enabled = False

    @classmethod
    def set_predicate(cls, predicate) -> None:
        """ Globally sets the Timer's Predicate

        The Predicate is used to determine whether a specific function that is
        decorated with @Timer will be timed or not.
        """
        cls._predicate = predicate
