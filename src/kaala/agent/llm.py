"""LLM client using Ollama."""

from abc import ABC, abstractmethod
from typing import Generator, Type
import asyncio
import json

from ollama import Client
from pydantic import BaseModel

from kaala.utils.model_parser import model_select


class BaseAgent(ABC):
    """Base Class for all the agents"""

    def __init__(
        self,
        system_prompt: str | None,
        model: str = "GLM-5-CLOUD",
        response_template: Type[BaseModel] | None = None,
    ):
        self.model = model_select(model)
        self.history: list[dict[str, str]] = []
        self.client = Client()

        schema_instruction = ""
        if response_template is not None:
            schema = response_template.model_json_schema()
            schema_instruction = (
                "\n\nYou MUST respond ONLY with a JSON array where each item matches this schema: "
                f"{json.dumps(schema, indent=2)}"
            )

        self.system_prompt = (system_prompt or "") + schema_instruction

    def _prepare_messages(
        self, prompt: str, context: list[dict] | None = None
    ) -> list[dict[str, str]]:
        """Build the message list for the Ollama chat API.

        Args:
            prompt: User's prompt.
            context: Optional list of {role, content} dicts from conversation history.

        Returns:
            List of messages for the Ollama chat API.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            for m in context:
                messages.append({"role": m["role"], "content": m["content"]})

        messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def chat(self, prompt: str, context: list[dict] | None = None, **overrides) -> str | None:
        """A function to maintain history and chat with prev msg context

        Args:
            prompt: User's prompt
            context: Optional conversation history

        Returns:
            str: Response of the llm to the user
        """
        messages = self._prepare_messages(prompt, context=context)
        options = overrides.pop("options", None)

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **overrides,
        )

        assistant_msg = response.message.content
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def generate(self, prompt: str, **overrides) -> str | None:
        """A function to generate immediate responses

        Args:
            prompt: User's prompt

        Returns:
            str: Response of the llm to the user
        """
        options = overrides.pop("options", None)

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            format="json",
            options=options,
            **overrides,
        )

        return response.response

    def chat_stream(self, prompt: str, **overrides) -> Generator[str | None, None, None]:
        """A function to maintain history and chat with prev msg context in a streaming manner

        Args:
            prompt: User's prompt

        Yields:
            Generator[str, None, None]: Streaming response of the llm to the user
        """
        messages = self._prepare_messages(prompt)
        options = overrides.pop("options", None)

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=options,
            **overrides,
        )

        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def generate_stream(
        self, prompt: str, **overrides
    ) -> Generator[str | None, None, None]:
        """A function to generate immediate responses in a streaming manner

        Args:
            prompt: User's prompt

        Yields:
            Generator[str, None, None]: Streaming response of the llm to the user
        """
        options = overrides.pop("options", None)

        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            format="json",
            stream=True,
            options=options,
            **overrides,
        )

        for chunk in stream:
            if chunk.response:
                yield chunk.response

    async def chat_async(self, prompt: str, context: list[dict] | None = None, **overrides: dict) -> str | None:
        """A function to maintain history and chat with previous message context asynchronously

        Args:
            prompt: User's prompt
            context: Optional conversation history

        Returns:
            str: Response of the llm to the user
        """
        messages = self._prepare_messages(prompt, context=context)
        options = overrides.pop("options", None)

        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=messages,
            options=options,
            **overrides,
        )

        assistant_msg = response.message.content
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    async def generate_async(self, prompt: str, **overrides: dict) -> str | None:
        """A function to generate immediate responses asynchronously

        Args:
            prompt: User's prompt

        Returns:
            str: Response of the llm to the user
        """
        options = overrides.pop("options", None)

        response = await asyncio.to_thread(
            self.client.generate,
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            format="json",
            options=options,
            **overrides,
        )
        return response.response

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