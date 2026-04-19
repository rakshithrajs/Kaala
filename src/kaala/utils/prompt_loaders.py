"""Loads System Prompts

Raises:
    FileError: File either does not exist or is moved from location

Returns:
    str: System Prompt
"""

from pathlib import Path

from kaala.utils.custom_errors import FileError

_SYSTEM_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SYSTEM_PROMPTS = {
    "Iccha": _SYSTEM_PROMPTS_DIR / "iccha.txt",
    "Karya": _SYSTEM_PROMPTS_DIR / "karya.txt",
    "Niyati": _SYSTEM_PROMPTS_DIR / "niyati.txt",
    "Karma": _SYSTEM_PROMPTS_DIR / "karma.txt",
}


def load_prompt(prompt: str) -> str:
    """Loads System Prompts from the text files

    Args:
        prompt (str): Prompt name that you wanna load

    Raises:
        FileError: File not found or location is changed

    Returns:
        str: The content of the system prompt
    """
    path = _SYSTEM_PROMPTS.get(prompt)
    if path is None:
        raise FileError(f"Unknown prompt: {prompt}")

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileError("File is missing or moved from the location") from exc