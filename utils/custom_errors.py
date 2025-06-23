"""A custom File error"""

class FileError(Exception):
    """Exception for error in loading files

    Args:
        Exception (FileNotFound): File either is missing or moved from location
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
