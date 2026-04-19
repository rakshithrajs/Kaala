"""Core LLM agent wrapper utilities."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Generator

from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.model_parser import model_select

load_dotenv(".env")

_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


class BaseAgent(ABC):
    """Base class for all personas."""

    def __init__(self, system_prompt: str | None, model: str = "GEMINI-1.5-PRO"):
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )
        self.model = model_select(model)
        self.client = genai.Client()

    def _prepare_prompt(self, prompt: str) -> str:
        return prompt

    @staticmethod
    def _parse_json_response(text: str | None) -> Any:
        """Parse potentially noisy LLM output into JSON if possible."""
        if not text:
            return None
        raw = text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(raw)
            if not match:
                return None
            return json.loads(match.group(1))

    def chat(self, prompt: str, **overrides) -> str | None:
        prompt = self._prepare_prompt(prompt)
        chat = self.client.chats.create(model=self.model, config=self.config, **overrides)
        return chat.send_message(prompt).text

    def generate(self, prompt: str, **overrides) -> str | None:
        prompt = self._prepare_prompt(prompt)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.config,
            **overrides,
        )
        return response.text

    async def generate_async(self, prompt: str, **overrides) -> str | None:
        prompt = self._prepare_prompt(prompt)
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=self.config,
            **overrides,
        )
        return response.text

    def generate_json(self, prompt: str, **overrides) -> Any:
        return self._parse_json_response(self.generate(prompt, **overrides))

    async def generate_async_json(self, prompt: str, **overrides) -> Any:
        return self._parse_json_response(await self.generate_async(prompt, **overrides))

    def generate_stream(self, prompt: str, **overrides) -> Generator[str | None, None, None]:
        prompt = self._prepare_prompt(prompt)
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=self.config,
            **overrides,
        )
        for chunk in response:
            yield chunk.text

    @abstractmethod
    def name(self) -> str:
        """Display name of the persona."""

    def info(self) -> dict[str, str]:
        return {"name": self.name(), "model": self.model}
