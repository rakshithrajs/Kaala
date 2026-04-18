"""Tests for storage repositories."""

import asyncio
import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from storage.database import Base
from storage.repositories import (
    GoalRepository,
    ScheduleRepository,
    HistoryRepository,
    UserContextRepository,
)
from storage.models import Goal, ScheduledPrompt, ConversationHistory, UserContext


@pytest_asyncio.fixture
async def session():
    """Create an in-memory SQLite session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_goal_repo_create_and_get(session: AsyncSession):
    repo = GoalRepository(session)
    goal = await repo.create(goal_text="Learn Python", details="Daily practice", status="pending")
    assert goal.id is not None
    assert goal.goal_text == "Learn Python"

    fetched = await repo.get_by_id(goal.id)
    assert fetched is not None
    assert fetched.goal_text == "Learn Python"


@pytest.mark.asyncio
async def test_goal_repo_get_pending(session: AsyncSession):
    repo = GoalRepository(session)
    await repo.create(goal_text="Goal 1", status="pending")
    await repo.create(goal_text="Goal 2", status="completed")

    pending = await repo.get_pending()
    assert len(pending) == 1
    assert pending[0].goal_text == "Goal 1"


@pytest.mark.asyncio
async def test_goal_repo_update_status(session: AsyncSession):
    repo = GoalRepository(session)
    goal = await repo.create(goal_text="Test goal", status="pending")
    updated = await repo.update_status(goal.id, "completed")
    assert updated is not None
    assert updated.status == "completed"
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_schedule_repo_create_and_get_due(session: AsyncSession):
    from datetime import datetime, timedelta

    schedule_repo = ScheduleRepository(session)
    goal_repo = GoalRepository(session)

    goal = await goal_repo.create(goal_text="Test goal")
    past_time = datetime.utcnow() - timedelta(hours=1)
    prompt = await schedule_repo.create(
        prompt_text="Check on goal",
        scheduled_for=past_time,
        goal_id=goal.id,
        prompt_type="check_in",
    )
    assert prompt.id is not None

    due = await schedule_repo.get_due()
    assert len(due) == 1
    assert due[0].prompt_text == "Check on goal"


@pytest.mark.asyncio
async def test_schedule_repo_mark_executed(session: AsyncSession):
    from datetime import datetime, timedelta

    repo = ScheduleRepository(session)
    past_time = datetime.utcnow() - timedelta(minutes=5)
    prompt = await repo.create(prompt_text="Test prompt", scheduled_for=past_time)

    executed = await repo.mark_executed(prompt.id)
    assert executed is not None
    assert executed.status == "executed"
    assert executed.executed_at is not None


@pytest.mark.asyncio
async def test_history_repo_save_and_get(session: AsyncSession):
    repo = HistoryRepository(session)
    await repo.save(agent_name="Niyati", role="assistant", content="Routing to Iccha")
    await repo.save(agent_name="Iccha", role="assistant", content="Goal detected")

    recent = await repo.get_recent(limit=10)
    assert len(recent) == 2

    iccha_entries = await repo.get_by_agent("Iccha")
    assert len(iccha_entries) == 1
    assert iccha_entries[0].agent_name == "Iccha"


@pytest.mark.asyncio
async def test_history_repo_context_for_agent(session: AsyncSession):
    repo = HistoryRepository(session)
    await repo.save(agent_name="Iccha", role="user", content="I want to learn guitar")
    await repo.save(agent_name="Iccha", role="assistant", content="Goal detected: learn guitar")

    context = await repo.get_context_for_agent("Iccha", limit=5)
    assert len(context) == 2
    assert context[0]["role"] == "user"


@pytest.mark.asyncio
async def test_user_context_repo_set_and_get(session: AsyncSession):
    repo = UserContextRepository(session)
    await repo.set("timezone", "Asia/Kolkata")
    value = await repo.get("timezone")
    assert value == "Asia/Kolkata"


@pytest.mark.asyncio
async def test_user_context_repo_update(session: AsyncSession):
    repo = UserContextRepository(session)
    await repo.set("key", "value1")
    await repo.set("key", "value2")
    value = await repo.get("key")
    assert value == "value2"


@pytest.mark.asyncio
async def test_user_context_repo_get_all(session: AsyncSession):
    repo = UserContextRepository(session)
    await repo.set("a", "1")
    await repo.set("b", "2")
    all_ctx = await repo.get_all()
    assert all_ctx == {"a": "1", "b": "2"}


@pytest.mark.asyncio
async def test_user_context_repo_delete(session: AsyncSession):
    repo = UserContextRepository(session)
    await repo.set("to_delete", "value")
    deleted = await repo.delete("to_delete")
    assert deleted is True
    value = await repo.get("to_delete")
    assert value is None