from .custom_errors import FileError


def load_prompt(prompt: str) -> str:
    system_prompts = {
        "goal": r"system_prompts\ICCHA.txt",
        "plan": r"system_prompts\KARYA.txt",
        "orchestrate": r"system_prompts\NIYATI.txt",
        "execute": r"system_prompts\KARMA.txt",
    }

    try:
        with open(system_prompts[prompt], "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileError("File is missing or moved from the location")
