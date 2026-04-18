"""Parses Model names

Raises:
    FileError: File not found

Returns:
    str: model name for the LLM provider
"""

import json
from pathlib import Path

from utils.custom_errors import FileError

_MODEL_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"


def model_select(model_name: str) -> str:
    """Select the model identifier for a given friendly name.

    Args:
        model_name (str): Friendly model name (e.g. "GEMINI-2-FLASH-LITE")

    Raises:
        FileError: Model config file does not exist

    Returns:
        str: The provider-specific model identifier
    """
    try:
        models: dict = json.loads(_MODEL_FILE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileError(
            "Model json file either does not exist or has been moved"
        ) from exc

    return models[model_name]
