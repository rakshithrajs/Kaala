"""A custom File error"""

from typing import Self

class FileError(Exception):
    """Exception for error in loading files

    Args:
        Exception (FileNotFound): File either is missing or moved from location
    """

    def __init__(self: Self, message: str) -> None:
        self.message = message
        super().__init__(message)
