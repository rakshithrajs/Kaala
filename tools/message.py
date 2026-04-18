"""Message tool — sends an immediate message to the user."""

from typing import Any

from core.tools import Tool


class MessageTool(Tool):
    """Send an immediate message/check-in to the user."""

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Parameters: text (str)"

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        text = parameters.get("text", "")
        if not text:
            return {"success": False, "error": "Missing 'text' parameter"}

        return {"success": True, "result": text}