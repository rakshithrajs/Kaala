"""Tests for the tool framework."""

import pytest
import pytest_asyncio

from kaala.core.tools import ToolRegistry
from kaala.tools.reminder import ReminderTool
from kaala.tools.message import MessageTool


def test_tool_registry_register():
    registry = ToolRegistry()
    registry.register(ReminderTool())
    registry.register(MessageTool())

    assert "reminder" in registry.list_tools()
    assert "message" in registry.list_tools()


def test_tool_registry_get():
    registry = ToolRegistry()
    registry.register(ReminderTool())

    tool = registry.get("reminder")
    assert tool is not None
    assert tool.name == "reminder"

    missing = registry.get("nonexistent")
    assert missing is None


def test_tool_registry_descriptions():
    registry = ToolRegistry()
    registry.register(ReminderTool())
    registry.register(MessageTool())

    desc = registry.tool_descriptions()
    assert "reminder" in desc
    assert "message" in desc


@pytest.mark.asyncio
async def test_message_tool():
    tool = MessageTool()
    result = await tool.execute({"text": "Hello world"})
    assert result["success"] is True
    assert result["result"] == "Hello world"


@pytest.mark.asyncio
async def test_message_tool_missing_text():
    tool = MessageTool()
    result = await tool.execute({})
    assert result["success"] is False
    assert "text" in result["error"]


@pytest.mark.asyncio
async def test_reminder_tool_missing_params():
    tool = ReminderTool()
    result = await tool.execute({})
    assert result["success"] is False
    assert "Missing" in result["error"]


@pytest.mark.asyncio
async def test_reminder_tool_invalid_timestamp():
    tool = ReminderTool()
    result = await tool.execute({"message": "Test", "time": "not-a-timestamp"})
    assert result["success"] is False
    assert "Invalid" in result["error"]