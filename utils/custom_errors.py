"""A custom File error"""


class FileError(Exception):
    """Exception for error in loading files

    Args:
        Exception (FileNotFound): File either is missing or moved from location
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AgentError(Exception):
    """Agent error

    Args:
        Exception (class): Inherits the Exception class to make a custom error
    """

    def __init__(
        self,
        message: str = """Agent entered does not exist. Available options:
        1. niyati
        2. iccha
        3. karya
        4. karma
        5. normal""",
    ) -> None:
        self.message = message
        super().__init__(message)
