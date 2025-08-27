"""Agent Factory"""

from agent.personas import Niyati, Iccha, Karya, Karma, Normal
from agent.llm import BaseAgent
from utils.custom_errors import AgentError


class AgentFactory:
    """A menu select agent with a simple string.
    options:
        1. niyati
        2. iccha
        3. karya
        4. karma
        5. normal

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    agents = {
        "niyati": Niyati,
        "iccha": Iccha,
        "karya": Karya,
        "karma": Karma,
        "normal": Normal,
    }

    @staticmethod
    def create(name: str = "normal", model:str = "GEMINI-1.5-PRO") -> BaseAgent:
        """Create an agent based on the specified string

        Args:
            name (str, optional): Type of agent you want. Defaults to "normal".

        Raises:
            AgentError: No specified agent

        Returns:
            BaseAgent: Type of agent you want
        """
        name = name.lower()
        try:
            return AgentFactory.agents[name](model)
        except KeyError as e:
            raise AgentError(
                f"No agent ('{name}'), Available agents: {','.join(AgentFactory.agents.keys())}"
            ) from e

    @staticmethod
    def list_agents() -> list[str]:
        """List all available agents

        Returns:
            list[str]: List of agent names
        """
        return list(AgentFactory.agents.keys())

    @staticmethod
    def get_agent_info(agent: BaseAgent) -> dict:
        """Get information about the specified agent

        Args:
            name (str, optional): Type of agent you want. Defaults to "normal".

        Returns:
            dict: Information about the agent
        """
        return agent.info()
