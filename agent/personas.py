"""Contains all the personas"""

from datetime import datetime

from agent.llm import BaseAgent
from utils.prompt_loaders import load_prompt


class Niyati(BaseAgent):
    """Niyati - The Orchestrator

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self):
        super().__init__(system_prompt=load_prompt("Niyati"))

    def name(self):
        return "Niyati - The Orchestrator"


class Iccha(BaseAgent):
    """Iccha - The Goal Extractor

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self):
        super().__init__(system_prompt=load_prompt("Iccha"))

    def name(self):
        return "Iccha - The Goal Extractor"


class Karya(BaseAgent):
    """Karya - The Planner
    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self):
        super().__init__(system_prompt=load_prompt("Karya"))

    def _prepare_prompt(self, prompt: str) -> str:
        return f"Today's date and time is {datetime.now()} \n {prompt}"

    def name(self):
        return "Karya - The Planner"


class Karma(BaseAgent):
    """Karma - The Executor

    Args:
        BaseAgent (BaseAgent): Base class
    """

    def __init__(self):
        super().__init__(system_prompt=load_prompt("Karma"))

    def name(self):
        return "Karma - The Executor"


class Normal(BaseAgent):
    """Normal

    Args:
        BaseAgent (Class): Base Class
    """

    def __init__(self):
        super().__init__(system_prompt=None)

    def name(self):
        return "Normal"
