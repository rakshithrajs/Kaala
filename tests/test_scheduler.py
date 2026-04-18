"""Tests for the prompt scheduler."""

import pytest
from unittest.mock import MagicMock

from core.scheduler import PromptScheduler


def test_scheduler_init():
    mock_orchestrator = MagicMock()
    scheduler = PromptScheduler(orchestrator=mock_orchestrator, poll_interval=30)
    assert scheduler.poll_interval == 30
    assert scheduler.orchestrator is mock_orchestrator


def test_scheduler_default_interval():
    mock_orchestrator = MagicMock()
    scheduler = PromptScheduler(orchestrator=mock_orchestrator)
    assert scheduler.poll_interval == 60


def test_should_suppress_non_check_in():
    """Non-check_in prompt types should never be suppressed."""
    mock_orchestrator = MagicMock()
    scheduler = PromptScheduler(orchestrator=mock_orchestrator)

    # _should_suppress is async but for non-check_in returns False synchronously
    import asyncio
    assert asyncio.get_event_loop().run_until_complete(
        scheduler._should_suppress("reminder")
    ) is False
    assert asyncio.get_event_loop().run_until_complete(
        scheduler._should_suppress("follow_up")
    ) is False