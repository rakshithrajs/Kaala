"""Black-box tests for API endpoints and WebSocket."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)

from kaala.storage.database import Base, Database
from kaala.storage.models import Goal, ScheduledPrompt, ConversationHistory, UserContext
from kaala.storage.repositories import (
    GoalRepository,
    ScheduleRepository,
    HistoryRepository,
    UserContextRepository,
)
from kaala.web.websocket import ConnectionManager


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(db_engine):
    """Create an async session for testing."""
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def db(db_engine):
    """Create a Database instance for testing."""
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    db_instance = Database.__new__(Database)
    db_instance.engine = db_engine
    db_instance.async_session = session_factory
    return db_instance


# ─── Goals API Tests ─────────────────────────────────────────────────────────


class TestGoalsAPI:
    """Test goals CRUD endpoints via repository (black-box from API perspective)."""

    @pytest.mark.asyncio
    async def test_create_goal(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="Learn testing", details="Write unit tests")
        assert goal.id is not None
        assert goal.goal_text == "Learn testing"
        assert goal.details == "Write unit tests"
        assert goal.status == "pending"

    @pytest.mark.asyncio
    async def test_get_goal_by_id(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="Test goal")
        fetched = await repo.get_by_id(goal.id)
        assert fetched is not None
        assert fetched.goal_text == "Test goal"

    @pytest.mark.asyncio
    async def test_get_nonexistent_goal(self, session):
        repo = GoalRepository(session)
        result = await repo.get_by_id(9999)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all_goals(self, session):
        repo = GoalRepository(session)
        await repo.create(goal_text="Goal 1")
        await repo.create(goal_text="Goal 2")
        goals = await repo.get_all()
        assert len(goals) == 2

    @pytest.mark.asyncio
    async def test_filter_goals_by_status(self, session):
        repo = GoalRepository(session)
        await repo.create(goal_text="Pending", status="pending")
        await repo.create(goal_text="Completed", status="completed")
        pending = await repo.get_all(status="pending")
        assert len(pending) == 1
        assert pending[0].goal_text == "Pending"

    @pytest.mark.asyncio
    async def test_update_goal_status_to_completed(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="To complete")
        updated = await repo.update_status(goal.id, "completed")
        assert updated.status == "completed"
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_goal_status_to_cancelled(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="To cancel")
        updated = await repo.update_status(goal.id, "cancelled")
        assert updated.status == "cancelled"
        assert updated.completed_at is None

    @pytest.mark.asyncio
    async def test_update_nonexistent_goal_status(self, session):
        repo = GoalRepository(session)
        result = await repo.update_status(9999, "completed")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_goal_details(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="Goal", details="Old details")
        updated = await repo.update_details(goal.id, "New details")
        assert updated.details == "New details"

    @pytest.mark.asyncio
    async def test_delete_goal(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="To delete")
        result = await repo.delete(goal.id)
        assert result is True
        assert await repo.get_by_id(goal.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_goal(self, session):
        repo = GoalRepository(session)
        result = await repo.delete(9999)
        assert result is False

    @pytest.mark.asyncio
    async def test_goal_defaults(self, session):
        repo = GoalRepository(session)
        goal = await repo.create(goal_text="Minimal goal")
        assert goal.status == "pending"
        assert goal.details is None
        assert goal.created_at is not None
        assert goal.completed_at is None


# ─── Schedule API Tests ──────────────────────────────────────────────────────


class TestScheduleAPI:
    """Test schedule CRUD via repository."""

    @pytest.mark.asyncio
    async def test_create_schedule(self, session):
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="Check on goal",
            scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
            prompt_type="check_in",
        )
        assert prompt.id is not None
        assert prompt.status == "pending"

    @pytest.mark.asyncio
    async def test_create_schedule_with_goal(self, session):
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)
        goal = await goal_repo.create(goal_text="Test goal")
        prompt = await schedule_repo.create(
            prompt_text="Follow up",
            scheduled_for=datetime.now(timezone.utc) + timedelta(days=1),
            goal_id=goal.id,
            prompt_type="follow_up",
        )
        assert prompt.goal_id == goal.id

    @pytest.mark.asyncio
    async def test_get_due_prompts(self, session):
        repo = ScheduleRepository(session)
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        await repo.create(prompt_text="Past due", scheduled_for=past_time)
        await repo.create(prompt_text="Future", scheduled_for=future_time)

        due = await repo.get_due()
        assert len(due) == 1
        assert due[0].prompt_text == "Past due"

    @pytest.mark.asyncio
    async def test_get_due_before_specific_time(self, session):
        repo = ScheduleRepository(session)
        t1 = datetime.now(timezone.utc) - timedelta(hours=2)
        t2 = datetime.now(timezone.utc) - timedelta(hours=1)
        await repo.create(prompt_text="Very past", scheduled_for=t1)
        await repo.create(prompt_text="Recent past", scheduled_for=t2)

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=90)
        due = await repo.get_due(before=cutoff)
        assert len(due) == 1

    @pytest.mark.asyncio
    async def test_mark_executed(self, session):
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="Test", scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        executed = await repo.mark_executed(prompt.id)
        assert executed.status == "executed"
        assert executed.executed_at is not None

    @pytest.mark.asyncio
    async def test_mark_failed(self, session):
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="Test", scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        failed = await repo.mark_failed(prompt.id)
        assert failed.status == "failed"

    @pytest.mark.asyncio
    async def test_cancel_prompt(self, session):
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="Test", scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        cancelled = await repo.cancel(prompt.id)
        assert cancelled.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, session):
        repo = ScheduleRepository(session)
        result = await repo.cancel(9999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_goal(self, session):
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)
        goal = await goal_repo.create(goal_text="Test")
        await schedule_repo.create(prompt_text="P1", scheduled_for=datetime.now(timezone.utc), goal_id=goal.id)
        await schedule_repo.create(prompt_text="P2", scheduled_for=datetime.now(timezone.utc), goal_id=goal.id)
        prompts = await schedule_repo.get_by_goal(goal.id)
        assert len(prompts) == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self, session):
        repo = ScheduleRepository(session)
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        await repo.create(prompt_text="P1", scheduled_for=past, status="pending")
        executed_prompt = await repo.create(prompt_text="P2", scheduled_for=past)
        await repo.mark_executed(executed_prompt.id)

        pending = await repo.get_all(status="pending")
        executed = await repo.get_all(status="executed")
        assert len(pending) == 1
        assert len(executed) == 1

    @pytest.mark.asyncio
    async def test_delete_schedule(self, session):
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="To delete", scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        result = await repo.delete(prompt.id)
        assert result is True
        assert await repo.get_by_id(prompt.id) is None

    @pytest.mark.asyncio
    async def test_prompt_type_values(self, session):
        repo = ScheduleRepository(session)
        for ptype in ("check_in", "reminder", "follow_up"):
            prompt = await repo.create(
                prompt_text=f"Type {ptype}",
                scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
                prompt_type=ptype,
            )
            assert prompt.prompt_type == ptype


# ─── History API Tests ──────────────────────────────────────────────────────


class TestHistoryAPI:
    """Test history repository operations."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, session):
        repo = HistoryRepository(session)
        entry = await repo.save(agent_name="Niyati", role="assistant", content="Routed to Iccha")
        assert entry.id is not None
        assert entry.agent_name == "Niyati"

    @pytest.mark.asyncio
    async def test_get_recent_ordered(self, session):
        repo = HistoryRepository(session)
        await repo.save(agent_name="User", role="user", content="First")
        await repo.save(agent_name="User", role="user", content="Second")
        entries = await repo.get_recent(limit=10)
        assert len(entries) == 2
        # Should be in chronological order (oldest first)
        assert entries[0].content == "First"
        assert entries[1].content == "Second"

    @pytest.mark.asyncio
    async def test_filter_by_agent(self, session):
        repo = HistoryRepository(session)
        await repo.save(agent_name="Niyati", role="assistant", content="Route")
        await repo.save(agent_name="Iccha", role="assistant", content="Goal")
        niyati = await repo.get_by_agent("Niyati")
        assert len(niyati) == 1
        assert niyati[0].agent_name == "Niyati"

    @pytest.mark.asyncio
    async def test_limit(self, session):
        repo = HistoryRepository(session)
        for i in range(5):
            await repo.save(agent_name="User", role="user", content=f"Msg {i}")
        entries = await repo.get_recent(limit=3)
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_clear_all(self, session):
        repo = HistoryRepository(session)
        await repo.save(agent_name="A", role="user", content="test")
        await repo.save(agent_name="B", role="user", content="test")
        count = await repo.clear()
        assert count == 2

    @pytest.mark.asyncio
    async def test_clear_specific_agent(self, session):
        repo = HistoryRepository(session)
        await repo.save(agent_name="Niyati", role="assistant", content="Route")
        await repo.save(agent_name="Iccha", role="assistant", content="Goal")
        count = await repo.clear(agent_name="Niyati")
        assert count == 1
        remaining = await repo.get_recent()
        assert len(remaining) == 1
        assert remaining[0].agent_name == "Iccha"

    @pytest.mark.asyncio
    async def test_context_for_agent(self, session):
        repo = HistoryRepository(session)
        await repo.save(agent_name="Iccha", role="user", content="I want to learn guitar")
        await repo.save(agent_name="Iccha", role="assistant", content="Goal detected")
        context = await repo.get_context_for_agent("Iccha", limit=5)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


# ─── UserContext API Tests ──────────────────────────────────────────────────


class TestUserContextAPI:
    @pytest.mark.asyncio
    async def test_set_and_get(self, session):
        repo = UserContextRepository(session)
        await repo.set("timezone", "Asia/Kolkata")
        value = await repo.get("timezone")
        assert value == "Asia/Kolkata"

    @pytest.mark.asyncio
    async def test_upsert(self, session):
        repo = UserContextRepository(session)
        await repo.set("key", "v1")
        await repo.set("key", "v2")
        assert await repo.get("key") == "v2"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, session):
        repo = UserContextRepository(session)
        assert await repo.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_all(self, session):
        repo = UserContextRepository(session)
        await repo.set("a", "1")
        await repo.set("b", "2")
        all_ctx = await repo.get_all()
        assert all_ctx == {"a": "1", "b": "2"}

    @pytest.mark.asyncio
    async def test_delete_existing(self, session):
        repo = UserContextRepository(session)
        await repo.set("to_delete", "value")
        assert await repo.delete("to_delete") is True
        assert await repo.get("to_delete") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, session):
        repo = UserContextRepository(session)
        assert await repo.delete("nonexistent") is False


# ─── WebSocket ConnectionManager Tests ──────────────────────────────────────


class TestConnectionManager:
    def test_init(self):
        mgr = ConnectionManager()
        assert mgr.active_connections == []

    @pytest.mark.asyncio
    async def test_connect(self):
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect(mock_ws)
        assert mock_ws in mgr.active_connections
        mock_ws.accept.assert_called_once()

    def test_disconnect(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mgr.active_connections.append(mock_ws)
        mgr.disconnect(mock_ws)
        assert mock_ws not in mgr.active_connections

    def test_disconnect_not_connected(self):
        mgr = ConnectionManager()
        mock_ws = MagicMock()
        mgr.disconnect(mock_ws)  # Should not raise

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr.active_connections = [ws1, ws2]
        await mgr.broadcast("test_event", {"key": "value"})
        expected = json.dumps({"event": "test_event", "data": {"key": "value"}})
        ws1.send_text.assert_called_once_with(expected)
        ws2.send_text.assert_called_once_with(expected)

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected(self):
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws2.send_text.side_effect = Exception("disconnected")
        mgr.active_connections = [ws1, ws2]
        await mgr.broadcast("event", {"data": True})
        assert ws2 not in mgr.active_connections
        assert ws1 in mgr.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_empty(self):
        mgr = ConnectionManager()
        await mgr.broadcast("event", {})  # No connections, should not raise


# ─── FastAPI Integration Tests ──────────────────────────────────────────────


class TestFastAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient.

    These test the routes against an in-memory DB with mocked Orchestrator.
    """

    @pytest.fixture
    def client(self):
        """Create a test client with in-memory DB and mocked orchestrator."""
        from fastapi import FastAPI
        from kaala.web.routes import chat, goals, schedules, history
        from kaala.web.dependencies import get_orchestrator, get_db_session

        test_app = FastAPI()
        test_app.include_router(chat.router, prefix="/api")
        test_app.include_router(goals.router, prefix="/api")
        test_app.include_router(schedules.router, prefix="/api")
        test_app.include_router(history.router, prefix="/api")

        mock_orch = AsyncMock()
        mock_orch.process_user_input = AsyncMock(
            return_value={"type": "conversation", "response": "Hello!", "signature": "Iccha"}
        )

        test_app.dependency_overrides[get_orchestrator] = lambda: mock_orch

        # Set up in-memory DB for the routes that use get_db_session
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        import asyncio

        async def setup_db():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            db = Database.__new__(Database)
            db.engine = engine
            db.async_session = session_factory
            return db

        db_instance = asyncio.get_event_loop().run_until_complete(setup_db())

        # Set app.state.db so get_db_session can access it
        test_app.state.db = db_instance

        with TestClient(test_app) as c:
            yield c

        asyncio.get_event_loop().run_until_complete(engine.dispose())

    def test_chat_endpoint_success(self, client):
        response = client.post("/api/chat", json={"message": "Hello!"})
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "conversation"
        assert data["response"] == "Hello!"

    def test_chat_endpoint_invalid_body(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_endpoint_wrong_field(self, client):
        response = client.post("/api/chat", json={"msg": "hello"})
        assert response.status_code == 422

    def test_goals_list_empty(self, client):
        response = client.get("/api/goals")
        assert response.status_code == 200
        assert response.json() == []

    def test_schedules_list_empty(self, client):
        response = client.get("/api/schedules")
        assert response.status_code == 200
        assert response.json() == []

    def test_history_list_empty(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_goal_not_found(self, client):
        response = client.get("/api/goals/9999")
        assert response.status_code == 404

    def test_goal_delete_not_found(self, client):
        response = client.delete("/api/goals/9999")
        assert response.status_code == 404

    def test_schedule_cancel_not_found(self, client):
        response = client.post("/api/schedules/9999/cancel")
        assert response.status_code == 404

    def test_goal_update_not_found(self, client):
        response = client.patch("/api/goals/9999", json={"status": "completed"})
        assert response.status_code == 404


# ─── GoalUpdate Schema Tests ────────────────────────────────────────────────


class TestGoalUpdateSchema:
    def test_both_fields(self):
        from kaala.web.routes.goals import GoalUpdate
        update = GoalUpdate(status="completed", details="Updated details")
        assert update.status == "completed"
        assert update.details == "Updated details"

    def test_status_only(self):
        from kaala.web.routes.goals import GoalUpdate
        update = GoalUpdate(status="cancelled")
        assert update.status == "cancelled"
        assert update.details is None

    def test_details_only(self):
        from kaala.web.routes.goals import GoalUpdate
        update = GoalUpdate(details="New details")
        assert update.status is None
        assert update.details == "New details"

    def test_empty_update(self):
        from kaala.web.routes.goals import GoalUpdate
        update = GoalUpdate()
        assert update.status is None
        assert update.details is None


# ─── Chat Request Schema Tests ──────────────────────────────────────────────


class TestChatRequestSchema:
    def test_valid_request(self):
        from kaala.web.routes.chat import ChatRequest
        req = ChatRequest(message="Hello!")
        assert req.message == "Hello!"

    def test_missing_message(self):
        from kaala.web.routes.chat import ChatRequest
        with pytest.raises(Exception):
            ChatRequest()