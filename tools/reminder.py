"""Reminder tool — schedules a prompt for future execution."""

from datetime import datetime
from typing import Any

from core.tools import Tool
from storage.database import get_db
from storage.repositories import ScheduleRepository


class ReminderTool(Tool):
    """Schedule a reminder for the user at a future time."""

    @property
    def name(self) -> str:
        return "reminder"

    @property
    def description(self) -> str:
        return "Schedule a reminder at a future time. Parameters: message (str), time (ISO timestamp str)"

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        message = parameters.get("message", "")
        time_str = parameters.get("time", "")

        if not message or not time_str:
            return {"success": False, "error": "Missing 'message' or 'time' parameter"}

        try:
            scheduled_for = datetime.fromisoformat(time_str.replace(" ", "T"))
        except (ValueError, AttributeError):
            return {"success": False, "error": f"Invalid timestamp format: {time_str}"}

        db = await get_db()
        async with db.async_session() as session:
            repo = ScheduleRepository(session)
            prompt = await repo.create(
                prompt_text=message,
                scheduled_for=scheduled_for,
                prompt_type="reminder",
            )

        return {
            "success": True,
            "result": f"Reminder scheduled for {scheduled_for.isoformat()}",
            "prompt_id": prompt.id,
        }