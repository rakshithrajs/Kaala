"""Utilities for loading system prompts."""

from pathlib import Path

from utils.custom_errors import FileError

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "system_prompts"


def load_prompt(prompt: str) -> str:
    system_prompts = {
        "Iccha": _PROMPTS_DIR / "ICCHA.txt",
        "Karya": _PROMPTS_DIR / "KARYA.txt",
        "Niyati": _PROMPTS_DIR / "NIYATI.txt",
        "Karma": _PROMPTS_DIR / "KARMA.txt",
    }

    try:
        with open(system_prompts[prompt], "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as exc:
        raise FileError("Prompt file is missing or moved from the location") from exc
    except KeyError as exc:
        raise FileError(f"Unknown prompt '{prompt}' requested") from exc
