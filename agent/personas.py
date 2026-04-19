"""Contains all the personas."""

from datetime import datetime

from agent.llm import BaseAgent
from utils.prompt_loaders import load_prompt


class Niyati(BaseAgent):
    def __init__(self, model: str):
        super().__init__(system_prompt=load_prompt("Niyati"), model=model)

    def name(self):
        return "Niyati - The Orchestrator"


class Iccha(BaseAgent):
    def __init__(self, model: str):
        super().__init__(system_prompt=load_prompt("Iccha"), model=model)

    def name(self):
        return "Iccha - The Goal Extractor"


class Karya(BaseAgent):
    def __init__(self, model: str):
        super().__init__(system_prompt=load_prompt("Karya"), model=model)

    def _prepare_prompt(self, prompt: str) -> str:
        return f"The current date and time is: {datetime.now().isoformat(sep=' ', timespec='microseconds')}\n{prompt}"

    def name(self):
        return "Karya - The Planner"


class Karma(BaseAgent):
    def __init__(self, model: str):
        super().__init__(system_prompt=load_prompt("Karma"), model=model)

    def name(self):
        return "Karma - The Executor"


class Normal(BaseAgent):
    def __init__(self, model: str):
        super().__init__(
            system_prompt=(
                'Give the response in JSON format: '
                '{"response": "<your response>", "signature": "Normal"}'
            ),
            model=model,
        )

    def name(self):
        return "Normal"
