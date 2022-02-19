""" Defines functions that remove most boilerplate related to file streams """


def reader(filename: str):
    """ Returns an open stream for reading

    :param filename: the name of the file
    :return: the open stream in 'r' mode
    """
    return open(filename, mode='r', encoding="utf-8")


def writer(filename: str):
    """ Returns an open stream for writing

    :param filename: the name of the file
    :return: the open stream in 'w' mode
    """
    return open(filename, mode='w', encoding="utf-8")
