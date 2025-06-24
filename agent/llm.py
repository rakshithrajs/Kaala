"""LLm.py"""

from abc import ABC, abstractmethod
from typing import Generator

import ollama

from utils.model_parser import model_select


class BaseAgent(ABC):
    """Base Class for all the agents"""

    def __init__(
        self,
        system_prompt: str | None = None,
        model: str = "LLAMA",
    ):
        self.system_prompt = system_prompt
        self.model = model_select(model)
        self.history = []

    def chat(self, prompt: str, **overides) -> ollama.ChatResponse:
        """A function to maintain history and chat with prev msg context

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.ChatResponse: Response of the llm to the user
        """
        # prompt = f"Today's date and time is {datetime.now()}" + prompt
        message = []
        self.history.append({"role": "user", "content": prompt})
        if self.system_prompt:
            message = [{"role": "system", "content": self.system_prompt}] + self.history
        else:
            message = self.history

        params = {
            "model": self.model,
            "messages": message,
            **overides,
        }

        response = ollama.chat(**params)

        self.history.append(
            {"role": "assistant", "content": response["message"]["content"]}
        )

        return response

    def generate(self, prompt: str, **overides) -> ollama.GenerateResponse:
        """A function to generate immidiate responses

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.GenerateResponse: Response of the llm to the user
        """

        # prompt = f"Today's date and time is {datetime.now()}" + prompt
        message = []
        if self.system_prompt:
            message.append({"role": "system", "content": self.system_prompt})
        message.append({"role": "user", "content": prompt})

        params = {
            "model": self.model,
            "messages": message,
            **overides,
        }

        return ollama.generate(**params)

    def chat_stream(
        self, prompt: str, **overides
    ) -> Generator[ollama.ChatResponse, None, None]:
        """A function to maintain history and chat with prev msg context in a streaming manner

        Args:
            prompt (str): User's prompt

        Yields:
            Generator[ollama.ChatResponse, None, None]: Streaming response of the llm to the user
        """
        message = []
        self.history.append({"role": "user", "content": prompt})
        if self.system_prompt:
            message = [{"role": "system", "content": self.system_prompt}] + self.history
        else:
            message = self.history

        params = {
            "model": self.model,
            "messages": message,
            **overides,
        }

        response = ollama.chat(**params, stream=True)

        for chunck in response:
            self.history.append(
                {"role": "assistant", "content": chunck["message"]["content"]}
            )
            yield chunck["message"]["content"]

    def reset(self):
        """Reset Conversation history"""
        self.history = []

    @abstractmethod
    def name(self) -> str:
        """Name of the persona

        Returns:
            str: Name of the persona
        """

    def info(self):
        """Returns Agent info

        Returns:
            dict: details about the agent like name, base model, system prompt
        """
        return {
            "name": self.name(),
            "model": self.model,
            "persona": (
                self.system_prompt[:50] + "..." if self.system_prompt else "None"
            ),
        }
