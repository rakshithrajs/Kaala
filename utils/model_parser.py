"""Parses Model names

Raises:
    FileError: File not found

Returns:
    str: model name for ollama to understand
"""

import json
from utils.custom_errors import FileError

MODEL_FILE_PATH = r"config\models.json"


def model_select(model_name: str) -> str:
    """A fucntion that selects the right model

    Args:
        model_name (str): Name of the model to be returned

    Raises:
        FileError: File does not exist or moved from location

    Returns:
        str: The correct name of the model for ollama to understand
    """
    try:
        with open(MODEL_FILE_PATH, "r", encoding="utf-8") as f:
            models: dict = json.loads(f.read())
    except FileNotFoundError as exc:
        raise FileError(
            "Model json file either does not exist or has been moved"
        ) from exc

    return models[model_name]
