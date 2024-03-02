"""
Module that provides several custom exceptions that can be thrown.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-24
"""


class GeneralException(Exception):
    """
    Unknown error caused a crash
    """

    pass


class InsufficientMemoryException(Exception):
    """
    System memory is not large enough to create a frame buffer
    """

    pass


class FileNotFoundException(Exception):
    """
    Requested file could not be found in file system
    """

    pass
