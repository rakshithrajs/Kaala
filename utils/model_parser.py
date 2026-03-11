"""Model name alias parser."""

import json
from pathlib import Path

from utils.custom_errors import FileError

MODEL_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"


def model_select(model_name: str) -> str:
    try:
        with open(MODEL_FILE_PATH, "r", encoding="utf-8") as file:
            models: dict[str, str] = json.load(file)
    except FileNotFoundError as exc:
        raise FileError("Model json file either does not exist or has been moved") from exc

    if model_name not in models:
        available = ", ".join(models.keys())
        raise FileError(f"Unknown model alias '{model_name}'. Available: {available}")

    return models[model_name]
