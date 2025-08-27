"""LLm.py"""

from abc import ABC, abstractmethod
from typing import Generator
import asyncio

from google import genai
from google.genai import types
from dotenv import load_dotenv

from utils.model_parser import model_select

load_dotenv(".env")


class BaseAgent(ABC):
    """Base Class for all the agents"""

    def __init__(
        self,
        system_prompt: str | None,
        model: str = "GEMINI-1.5-PRO",
    ):
        self.config = types.GenerateContentConfig(system_instruction=system_prompt)
        self.model = model_select(model)
        self.history = []
        self.client = genai.Client()

    def _prepare_prompt(self, prompt: str) -> str:
        """Prepare the prompt by adding system prompt if exists

        Args:
            pormpt (str): User's prompt

        Returns:
            str: Prepared prompt
        """
        return prompt

    def chat(self, prompt: str, **overides) -> str | None:
        """A function to maintain history and chat with prev msg context

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.ChatResponse: Response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "config": self.config,
            **overides,
        }

        chat = self.client.chats.create(**params)

        response = chat.send_message(prompt)

        return response.text

    def generate(self, prompt: str, **overides) -> str | None:
        """A function to generate immidiate responses

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.GenerateResponse: Response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "contents": prompt,
            "config": self.config,
            **overides,
        }

        return self.client.models.generate_content(**params).text

    def chat_stream(self, prompt: str, **overides) -> Generator[str | None, None, None]:
        """A function to maintain history and chat with prev msg context in a streaming manner

        Args:
            prompt (str): User's prompt

        Yields:
            Generator[ollama.ChatResponse, None, None]: Streaming response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "contents": prompt,
            "config": self.config,
            **overides,
        }

        chat = self.client.chats.create(**params)

        response = chat.send_message_stream(prompt)

        for chunk in response:
            yield chunk.text

    def generate_stream(
        self, prompt: str, **overides
    ) -> Generator[str | None, None, None]:
        """A function to generate immidiate responses in a streaming manner

        Args:
            prompt (str): User's prompt

        Yields:
            Generator[str, None, None]: Streaming response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "contents": prompt,
            "config": self.config,
            **overides,
        }

        response = self.client.models.generate_content_stream(**params)
        for chunk in response:
            yield chunk.text

    async def chat_async(self, prompt: str, **overides: dict) -> str | None:
        """A function to maintain history and chat with previous message context in an asynchronous
        manner

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.ChatResponse: Response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "config": self.config,
            **overides,
        }

        chat = self.client.chats.create(**params)
        response = await asyncio.to_thread(chat.send_message, prompt)

        return response.text

    async def generate_async(self, prompt: str, **overides: dict) -> str | None:
        """A function to generate immediate responses in an asynchronous manner

        Args:
            prompt (str): User's prompt

        Returns:
            ollama.GenerateResponse: Response of the llm to the user
        """
        prompt = self._prepare_prompt(prompt)

        params = {
            "model": self.model,
            "contents": prompt,
            "config": self.config,
            **overides,
        }

        response = await asyncio.to_thread(
            self.client.models.generate_content, **params
        )
        return response.text

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
        return {"name": self.name(), "model": self.model}
