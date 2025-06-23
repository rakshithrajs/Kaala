import json
from .custom_errors import FileError

MODEL_FILE_PATH = r"config\models.json"


def model_select(model_name: str) -> str:
    try:
        with open(MODEL_FILE_PATH, "r") as f:
            models: dict = json.loads(f.read())
    except FileNotFoundError:
        raise FileError("Model json file either does not exist or has been moved")

    return models[model_name]
