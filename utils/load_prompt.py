"""Loads System Prompts

Raises:
    FileError: File either does not exist or is moved from location

Returns:
    str: System Prompt
"""

from utils.custom_errors import FileError


def load_prompt(prompt: str) -> str:
    """Loads System Prompts from the text files

    Args:
        prompt (str): Prompt name that you wanna load

    Raises:
        FileError: File not found or location is changed

    Returns:
        str: The content of the system prompt
    """
    system_prompts = {
        "Iccha": r"system_prompts\ICCHA.txt",
        "Karya": r"system_prompts\KARYA.txt",
        "Niyati": r"system_prompts\NIYATI.txt",
        "Karma": r"system_prompts\KARMA.txt",
    }

    try:
        with open(system_prompts[prompt], "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileError("File is missing or moved from the location") from exc
