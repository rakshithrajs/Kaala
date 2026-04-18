"""Tool framework for Karma agent execution."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all Karma tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this tool does."""

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with the given parameters.

        Args:
            parameters: Tool-specific parameters.

        Returns:
            dict with at least a "success" key and a "result" or "error" key.
        """


class ToolRegistry:
    """Registry for Karma tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def tool_descriptions(self) -> str:
        """Return a formatted string of all tools for prompt inclusion."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)