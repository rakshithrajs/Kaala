"""Contains all the personas"""

from datetime import datetime

from kaala.agent.llm import BaseAgent
from kaala.utils.prompt_loaders import load_prompt
from kaala.utils.response_templates import (
    IcchaResponse,
    KarmaResponse,
    KaryaResponse,
    NiyatiResponse,
    NormalResponse,
)


class Niyati(BaseAgent):
    """Niyati - The Orchestrator

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self, model: str):
        super().__init__(
            system_prompt=load_prompt("Niyati"),
            model=model,
            response_template=NiyatiResponse,
        )

    def name(self):
        return "Niyati - The Orchestrator"


class Iccha(BaseAgent):
    """Iccha - The Goal Extractor

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self, model: str):
        super().__init__(
            system_prompt=load_prompt("Iccha"),
            model=model,
            response_template=IcchaResponse,
        )

    def name(self):
        return "Iccha - The Goal Extractor"


class Karya(BaseAgent):
    """Karya - The Planner
    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self, model: str):
        super().__init__(
            system_prompt=load_prompt("Karya"),
            model=model,
            response_template=KaryaResponse,
        )

    def _prepare_prompt(self, prompt: str, context: list[dict] | None = None) -> str:
        now = datetime.now().isoformat()
        if context:
            history_lines = "\n".join(
                f"{m['role']}: {m['content']}" for m in context
            )
            return f"Today's date and time is {now}\n\nPrevious conversation:\n{history_lines}\n\nCurrent input: {prompt}"
        return f"Today's date and time is {now}\n\nCurrent input: {prompt}"

    def name(self):
        return "Karya - The Planner"


class Karma(BaseAgent):
    """Karma - The Executor

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self, model: str):
        super().__init__(
            system_prompt=load_prompt("Karma"),
            model=model,
            response_template=KarmaResponse,
        )

    def name(self):
        return "Karma - The Executor"


class Normal(BaseAgent):
    """Normal

    Args:
        BaseAgent (Class): Base Class
    """

    def __init__(self, model: str):
        super().__init__(
            system_prompt="""Give the response in the following JSON format itself
            {"response" : <your response>, "signature" : "Normal"}""",
            model=model,
            response_template=NormalResponse,
        )

    def name(self):
        return "Normal"