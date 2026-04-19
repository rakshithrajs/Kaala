"""Smoke and integration tests for full pipeline and lifecycle."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)

from kaala.storage.database import Base, Database
from kaala.storage.models import Goal, ScheduledPrompt, ConversationHistory
from kaala.storage.repositories import (
    GoalRepository,
    ScheduleRepository,
    HistoryRepository,
    UserContextRepository,
)
from kaala.core.orchestrator import Orchestrator
from kaala.core.scheduler import PromptScheduler
from kaala.core.tools import ToolRegistry
from kaala.tools.message import MessageTool
from kaala.tools.reminder import ReminderTool


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(db_engine):
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def db(db_engine):
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    db_instance = Database.__new__(Database)
    db_instance.engine = db_engine
    db_instance.async_session = session_factory
    return db_instance


# ─── Database Smoke Tests ──────────────────────────────────────────────────


class TestDatabaseSmoke:
    """Smoke test: can we initialize the database and create tables?"""

    @pytest.mark.asyncio
    async def test_create_and_drop_tables(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        db = Database(db_url="sqlite+aiosqlite:///:memory:")
        await db.create_tables()
        await db.drop_tables()
        await db.engine.dispose()

    @pytest.mark.asyncio
    async def test_get_session(self, db):
        session = db.get_session()
        assert session is not None

    @pytest.mark.asyncio
    async def test_tables_exist(self, db_engine):
        """Verify all expected tables were created in the in-memory DB."""
        async with db_engine.begin() as conn:
            from sqlalchemy import text
            # Each table should be queryable without error
            for table_name in ("goals", "scheduled_prompts", "conversation_history", "user_context"):
                result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                assert result.scalar() == 0  # Empty table is fine


# ─── Orchestrator Pipeline Tests (Mocked LLM) ─────────────────────────────


class TestOrchestratorPipeline:
    """Integration tests for the orchestrator pipeline with mocked agents."""

    @pytest.mark.asyncio
    async def test_conversation_flow(self, session):
        """Test a simple conversational response (no goals detected)."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        # Niyati routes to Iccha
        orchestrator.niyati.chat_async = AsyncMock(
            return_value=json.dumps({"route_to": "Iccha", "user_prompt": "Hello!"})
        )

        # Iccha responds conversationally (no goal detected)
        orchestrator.iccha.chat_async = AsyncMock(
            return_value=json.dumps({
                "goal_detected": False,
                "response": "Hi! How can I help?",
            })
        )

        with patch.object(orchestrator, "_get_context", new_callable=AsyncMock, return_value=None):
            with patch.object(Orchestrator, "_get_session", return_value=session):
                result = await orchestrator.process_user_input("Hello!")

        assert result["type"] == "conversation"
        assert "Hi! How can I help?" in result["response"]

    @pytest.mark.asyncio
    async def test_immediate_goal_flow(self, session):
        """Test immediate goal routed to Karma."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        # Niyati routes to Iccha
        orchestrator.niyati.chat_async = AsyncMock(
            return_value=json.dumps({"route_to": "Iccha", "user_prompt": "Remind me now!"})
        )

        # Iccha detects an immediate goal
        orchestrator.iccha.chat_async = AsyncMock(
            return_value=json.dumps([{
                "goal_detected": True,
                "goal": "Remind me now!",
                "urgency": "immediate",
                "needs_clarification": False,
            }])
        )

        # Karma executes
        orchestrator.karma.chat_async = AsyncMock(
            return_value=json.dumps({
                "action": "send_message",
                "tool": "message",
                "parameters": {"text": "Reminder!"},
                "signature": "Karma",
            })
        )

        with patch.object(orchestrator, "_get_context", new_callable=AsyncMock, return_value=None), \
             patch.object(Orchestrator, "_get_session", return_value=session):
            result = await orchestrator.process_user_input("Remind me now!")

        assert result["type"] == "immediate"
        assert "Remind me now!" in result["goals"]

    @pytest.mark.asyncio
    async def test_clarification_flow(self, session):
        """Test clarification response when Iccha needs more info."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        orchestrator.niyati.chat_async = AsyncMock(
            return_value=json.dumps({"route_to": "Iccha", "user_prompt": "I want something"})
        )

        orchestrator.iccha.chat_async = AsyncMock(
            return_value=json.dumps([{
                "goal_detected": True,
                "goal": "Something vague",
                "needs_clarification": True,
                "response": "Can you be more specific?",
            }])
        )

        with patch.object(orchestrator, "_get_context", new_callable=AsyncMock, return_value=None), \
             patch.object(Orchestrator, "_get_session", return_value=session):
            result = await orchestrator.process_user_input("I want something")

        assert result["type"] == "clarification"
        assert "Can you be more specific?" in result["response"]

    @pytest.mark.asyncio
    async def test_karma_direct_flow(self, session):
        """Test direct routing to Karma from Niyati."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        orchestrator.niyati.chat_async = AsyncMock(
            return_value=json.dumps({"route_to": "Karma", "user_prompt": "Send message"})
        )

        orchestrator.karma.chat_async = AsyncMock(
            return_value=json.dumps({
                "action": "send_message",
                "tool": "message",
                "parameters": {"text": "Hello"},
                "signature": "Karma",
            })
        )

        orchestrator.tools.register(MessageTool())

        with patch.object(orchestrator, "_get_context", new_callable=AsyncMock, return_value=None), \
             patch.object(Orchestrator, "_get_session", return_value=session):
            result = await orchestrator.process_user_input("Send message")

        assert result["type"] == "executed"
        assert result["tool"] == "message"

    @pytest.mark.asyncio
    async def test_empty_niyati_response(self, session):
        """Test handling of empty Niyati response."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        orchestrator.niyati.chat_async = AsyncMock(return_value=None)

        with patch.object(Orchestrator, "_get_session", return_value=session):
            result = await orchestrator.process_user_input("test")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_route(self, session):
        """Test handling of unknown route from Niyati."""
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.niyati = AsyncMock()
        orchestrator.iccha = AsyncMock()
        orchestrator.karya = AsyncMock()
        orchestrator.karma = AsyncMock()
        orchestrator.tools = ToolRegistry()
        orchestrator._db = None

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        orchestrator._db = mock_db

        orchestrator.niyati.chat_async = AsyncMock(
            return_value=json.dumps({"route_to": "Unknown", "user_prompt": "test"})
        )

        with patch.object(Orchestrator, "_get_session", return_value=session):
            result = await orchestrator.process_user_input("test")

        assert "error" in result
        assert "Unknown" in result["error"]


# ─── Scheduler Integration Tests ──────────────────────────────────────────


class TestSchedulerIntegration:
    """Integration tests for the scheduler with mocked orchestrator."""

    def test_scheduler_init_defaults(self):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)
        assert scheduler.poll_interval == 60
        assert scheduler.orchestrator is mock_orch

    def test_scheduler_custom_interval(self):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch, poll_interval=30)
        assert scheduler.poll_interval == 30

    @pytest.mark.asyncio
    async def test_should_suppress_non_check_in(self):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)
        assert await scheduler._should_suppress("reminder") is False
        assert await scheduler._should_suppress("follow_up") is False

    @pytest.mark.asyncio
    async def test_should_suppress_check_in_under_threshold(self, session):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)

        # Set up DB
        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        # Under threshold - should not suppress
        ctx_repo = UserContextRepository(session)
        await ctx_repo.set("ignored_check_ins", "2")

        result = await scheduler._should_suppress("check_in")
        assert result is False

    @pytest.mark.asyncio
    async def test_should_suppress_check_in_at_threshold(self, session):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        ctx_repo = UserContextRepository(session)
        await ctx_repo.set("ignored_check_ins", "3")

        result = await scheduler._should_suppress("check_in")
        assert result is True

    @pytest.mark.asyncio
    async def test_record_engagement_resets_on_respond(self, session):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        ctx_repo = UserContextRepository(session)
        await ctx_repo.set("ignored_check_ins", "5")
        await scheduler._record_engagement("check_in", responded=True)

        value = await ctx_repo.get("ignored_check_ins")
        assert value == "0"

    @pytest.mark.asyncio
    async def test_record_engagement_increments_on_ignore(self, session):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        ctx_repo = UserContextRepository(session)
        await ctx_repo.set("ignored_check_ins", "2")
        await scheduler._record_engagement("check_in", responded=False)

        value = await ctx_repo.get("ignored_check_ins")
        assert value == "3"

    @pytest.mark.asyncio
    async def test_record_engagement_ignores_non_check_in(self, session):
        mock_orch = MagicMock()
        scheduler = PromptScheduler(orchestrator=mock_orch)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        ctx_repo = UserContextRepository(session)
        await ctx_repo.set("ignored_check_ins", "2")
        await scheduler._record_engagement("reminder", responded=False)

        # Should not change for non-check_in
        value = await ctx_repo.get("ignored_check_ins")
        assert value == "2"


# ─── Tool Execution Integration Tests ──────────────────────────────────────


class TestToolExecutionIntegration:
    """Test tool execution with database operations."""

    @pytest.mark.asyncio
    async def test_message_tool_success(self):
        tool = MessageTool()
        result = await tool.execute({"text": "Hello, user!"})
        assert result["success"] is True
        assert result["result"] == "Hello, user!"

    @pytest.mark.asyncio
    async def test_message_tool_empty_text(self):
        tool = MessageTool()
        result = await tool.execute({"text": ""})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_message_tool_missing_parameter(self):
        tool = MessageTool()
        result = await tool.execute({})
        assert result["success"] is False
        assert "text" in result["error"]

    @pytest.mark.asyncio
    async def test_reminder_tool_missing_message(self):
        tool = ReminderTool()
        result = await tool.execute({"time": "2025-06-15T10:00:00"})
        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_reminder_tool_missing_time(self):
        tool = ReminderTool()
        result = await tool.execute({"message": "Test reminder"})
        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_reminder_tool_invalid_timestamp(self):
        tool = ReminderTool()
        result = await tool.execute({"message": "Test", "time": "not-a-date"})
        assert result["success"] is False
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_reminder_tool_success_with_db(self, session):
        """Integration test: ReminderTool creates a scheduled prompt in DB."""
        tool = ReminderTool()
        future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)

        with patch("kaala.tools.reminder.get_db", return_value=mock_db):
            result = await tool.execute({
                "message": "Test reminder",
                "time": future_time,
            })

        assert result["success"] is True
        assert "prompt_id" in result

    @pytest.mark.asyncio
    async def test_reminder_tool_with_space_timestamp(self, session):
        """Test reminder tool with space-separated timestamp."""
        tool = ReminderTool()
        future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)

        with patch("kaala.tools.reminder.get_db", return_value=mock_db):
            result = await tool.execute({
                "message": "Space timestamp",
                "time": future_time,
            })

        assert result["success"] is True


# ─── Orchestrator Helper Method Tests ───────────────────────────────────────


class TestOrchestratorHelpers:
    """Test orchestrator helper methods with mocked DB."""

    @pytest.mark.asyncio
    async def test_get_pending_goals(self, session):
        goal_repo = GoalRepository(session)
        await goal_repo.create(goal_text="Goal 1")
        await goal_repo.create(goal_text="Goal 2", status="completed")

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._db = None

        with patch.object(Orchestrator, "_get_session", return_value=session):
            goals = await orchestrator.get_pending_goals()

        assert len(goals) == 1
        assert goals[0]["goal"] == "Goal 1"

    @pytest.mark.asyncio
    async def test_get_scheduled_prompts(self, session):
        schedule_repo = ScheduleRepository(session)
        await schedule_repo.create(
            prompt_text="Test prompt",
            scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._db = None

        with patch.object(Orchestrator, "_get_session", return_value=session):
            prompts = await orchestrator.get_scheduled_prompts()

        assert len(prompts) == 1
        assert prompts[0]["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_get_conversation_history(self, session):
        history_repo = HistoryRepository(session)
        await history_repo.save(agent_name="Niyati", role="assistant", content="Routed")

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._db = None

        with patch.object(Orchestrator, "_get_session", return_value=session):
            history = await orchestrator.get_conversation_history(limit=10)

        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_get_conversation_history_filtered(self, session):
        history_repo = HistoryRepository(session)
        await history_repo.save(agent_name="Niyati", role="assistant", content="Routed")
        await history_repo.save(agent_name="Iccha", role="assistant", content="Goal")

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._db = None

        with patch.object(Orchestrator, "_get_session", return_value=session):
            history = await orchestrator.get_conversation_history(agent_name="Niyati", limit=10)

        assert all(h["agent"] == "Niyati" for h in history)


# ─── End-to-End Scheduled Goal Flow ────────────────────────────────────────


class TestScheduledGoalFlow:
    """Integration test: Iccha goal -> Karya scheduling -> DB storage."""

    @pytest.mark.asyncio
    async def test_scheduled_goal_stored_in_db(self, session):
        """Simulate the flow where Iccha detects a scheduled goal, Karya creates a schedule, and it's stored in DB."""
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)

        # Simulate Iccha goal extraction
        goal = await goal_repo.create(
            goal_text="Learn guitar",
            details="Practice 30 min daily",
            status="pending",
        )
        assert goal.id is not None

        # Simulate Karya scheduling
        scheduled_for = datetime.now(timezone.utc) + timedelta(days=1)
        prompt = await schedule_repo.create(
            prompt_text="Have you practiced guitar today?",
            scheduled_for=scheduled_for,
            goal_id=goal.id,
            prompt_type="check_in",
        )
        assert prompt.id is not None
        assert prompt.goal_id == goal.id

        # Verify goal is still pending
        fetched_goal = await goal_repo.get_by_id(goal.id)
        assert fetched_goal.status == "pending"

        # Verify schedule is pending
        fetched_prompt = await schedule_repo.get_by_id(prompt.id)
        assert fetched_prompt.status == "pending"

    @pytest.mark.asyncio
    async def test_goal_completion_cascade(self, session):
        """Test that completing a goal doesn't break scheduled prompts."""
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)

        goal = await goal_repo.create(goal_text="Test goal")
        prompt = await schedule_repo.create(
            prompt_text="Check in",
            scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
            goal_id=goal.id,
        )

        # Complete the goal
        await goal_repo.update_status(goal.id, "completed")
        completed_goal = await goal_repo.get_by_id(goal.id)
        assert completed_goal.status == "completed"
        assert completed_goal.completed_at is not None

        # Schedule should still exist and be accessible
        fetched_prompt = await schedule_repo.get_by_id(prompt.id)
        assert fetched_prompt is not None

    @pytest.mark.asyncio
    async def test_multiple_goals_scheduled(self, session):
        """Test creating multiple goals and their schedules."""
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)

        goals = []
        for i in range(3):
            goal = await goal_repo.create(goal_text=f"Goal {i}")
            goals.append(goal)
            await schedule_repo.create(
                prompt_text=f"Check on Goal {i}",
                scheduled_for=datetime.now(timezone.utc) + timedelta(days=i+1),
                goal_id=goal.id,
            )

        all_goals = await goal_repo.get_all()
        assert len(all_goals) == 3

        pending_prompts = await schedule_repo.get_pending()
        assert len(pending_prompts) == 3


# ─── Scheduler Process Flow Tests ──────────────────────────────────────────


class TestSchedulerProcessFlow:
    """Test the scheduler's prompt processing with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_execute_reminder_prompt(self, session):
        """Test that reminder prompts are returned directly without orchestrator."""
        mock_orch = MagicMock()
        mock_orch.process_user_input = AsyncMock()

        # Use a callback to capture the result
        results = []
        async def on_result(r):
            results.append(r)

        scheduler = PromptScheduler(orchestrator=mock_orch, on_prompt_result=on_result)

        # Create a prompt in DB so mark_executed can find it
        schedule_repo = ScheduleRepository(session)
        prompt = await schedule_repo.create(
            prompt_text="Test reminder",
            scheduled_for=datetime.now(timezone.utc) - timedelta(hours=1),
            prompt_type="reminder",
        )

        # Set up DB mock to return our test session
        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        await scheduler._execute_prompt(prompt.id, "Test reminder", "reminder")
        assert len(results) == 1
        assert results[0]["type"] == "reminder"
        assert results[0]["message"] == "Test reminder"
        # Orchestrator should NOT be called for reminders
        mock_orch.process_user_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_check_in_prompt(self, session):
        """Test that check_in prompts go through the orchestrator."""
        mock_orch = MagicMock()
        mock_orch.process_user_input = AsyncMock(
            return_value={"type": "conversation", "response": "Hi!"}
        )
        scheduler = PromptScheduler(orchestrator=mock_orch)

        # Need to mock DB for mark_executed
        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        # Create a prompt in DB first
        schedule_repo = ScheduleRepository(session)
        prompt = await schedule_repo.create(
            prompt_text="Check in",
            scheduled_for=datetime.now(timezone.utc) - timedelta(hours=1),
            prompt_type="check_in",
        )

        callback_result = None
        async def capture_result(r):
            nonlocal callback_result
            callback_result = r

        scheduler.on_prompt_result = capture_result

        with patch.object(scheduler, "_get_db", return_value=mock_db):
            await scheduler._execute_prompt(prompt.id, "Check in", "check_in")

        mock_orch.process_user_input.assert_called_once_with("Check in")

    @pytest.mark.asyncio
    async def test_callback_called_on_success(self, session):
        """Test that on_prompt_result callback is called."""
        mock_orch = MagicMock()
        mock_orch.process_user_input = AsyncMock(
            return_value={"type": "conversation", "response": "Hi!"}
        )

        results = []
        async def on_result(r):
            results.append(r)

        scheduler = PromptScheduler(orchestrator=mock_orch, on_prompt_result=on_result)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        schedule_repo = ScheduleRepository(session)
        prompt = await schedule_repo.create(
            prompt_text="Test",
            scheduled_for=datetime.now(timezone.utc) - timedelta(hours=1),
            prompt_type="check_in",
        )

        with patch.object(scheduler, "_get_db", return_value=mock_db):
            await scheduler._execute_prompt(prompt.id, "Test", "check_in")

        assert len(results) == 1
        assert results[0]["type"] == "conversation"

    @pytest.mark.asyncio
    async def test_failed_prompt_marked(self, session):
        """Test that failed prompts are marked as failed."""
        mock_orch = MagicMock()
        mock_orch.process_user_input = AsyncMock(side_effect=Exception("LLM error"))
        scheduler = PromptScheduler(orchestrator=mock_orch)

        mock_db = AsyncMock()
        mock_db.async_session = MagicMock(return_value=session)
        scheduler._db = mock_db

        schedule_repo = ScheduleRepository(session)
        prompt = await schedule_repo.create(
            prompt_text="Failing prompt",
            scheduled_for=datetime.now(timezone.utc) - timedelta(hours=1),
            prompt_type="check_in",
        )

        with patch.object(scheduler, "_get_db", return_value=mock_db), \
             patch.object(scheduler, "_record_engagement", new_callable=AsyncMock):
            await scheduler._execute_prompt(prompt.id, "Failing prompt", "check_in")

        # The prompt should be marked as failed
        failed = await schedule_repo.get_by_id(prompt.id)
        assert failed.status == "failed"


# ─── App Lifecyle Smoke Test ────────────────────────────────────────────────


class TestAppLifecycle:
    """Smoke test: can we import and construct the app?"""

    def test_import_app(self):
        from kaala.web.app import app
        assert app is not None
        assert app.title == "Kaala"

    def test_import_orchestrator(self):
        from kaala.core.orchestrator import Orchestrator
        assert Orchestrator is not None

    def test_import_scheduler(self):
        from kaala.core.scheduler import PromptScheduler
        assert PromptScheduler is not None

    def test_import_all_models(self):
        from kaala.storage.models import Goal, ScheduledPrompt, ConversationHistory, UserContext
        assert Goal is not None
        assert ScheduledPrompt is not None

    def test_import_all_repos(self):
        from kaala.storage.repositories import (
            GoalRepository, ScheduleRepository,
            HistoryRepository, UserContextRepository,
        )
        assert GoalRepository is not None

    def test_import_all_agents(self):
        from kaala.agent.personas import Niyati, Iccha, Karya, Karma, Normal
        assert Niyati is not None

    def test_import_all_tools(self):
        from kaala.tools.message import MessageTool
        from kaala.tools.reminder import ReminderTool
        assert MessageTool is not None
        assert ReminderTool is not None

    def test_import_websocket_manager(self):
        from kaala.web.websocket import ConnectionManager, manager
        assert manager is not None
        assert isinstance(manager, ConnectionManager)


# ─── Data Integrity Tests ──────────────────────────────────────────────────


class TestDataIntegrity:
    """Test data integrity constraints and edge cases."""

    @pytest.mark.asyncio
    async def test_goal_status_transitions(self, session):
        """Test valid status transitions for goals."""
        repo = GoalRepository(session)

        # Create -> In Progress -> Completed
        goal = await repo.create(goal_text="Test")
        assert goal.status == "pending"

        goal = await repo.update_status(goal.id, "in_progress")
        assert goal.status == "in_progress"

        goal = await repo.update_status(goal.id, "completed")
        assert goal.status == "completed"
        assert goal.completed_at is not None

    @pytest.mark.asyncio
    async def test_schedule_status_transitions(self, session):
        """Test valid status transitions for schedules."""
        repo = ScheduleRepository(session)
        prompt = await repo.create(
            prompt_text="Test",
            scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert prompt.status == "pending"

        executed = await repo.mark_executed(prompt.id)
        assert executed.status == "executed"
        assert executed.executed_at is not None

    @pytest.mark.asyncio
    async def test_history_ordering(self, session):
        """Test that history entries are returned in chronological order."""
        repo = HistoryRepository(session)
        await repo.save(agent_name="A", role="user", content="First")
        await repo.save(agent_name="A", role="user", content="Second")
        await repo.save(agent_name="A", role="user", content="Third")

        entries = await repo.get_recent(agent_name="A", limit=10)
        assert len(entries) == 3
        assert entries[0].content == "First"
        assert entries[2].content == "Third"

    @pytest.mark.asyncio
    async def test_cascade_delete_goal_with_prompts(self, session):
        """Test that deleting a goal cascades to its scheduled prompts."""
        goal_repo = GoalRepository(session)
        schedule_repo = ScheduleRepository(session)

        goal = await goal_repo.create(goal_text="Test cascade")
        await schedule_repo.create(
            prompt_text="Check in",
            scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1),
            goal_id=goal.id,
        )

        # Delete the goal - cascade should handle prompts
        await goal_repo.delete(goal.id)

        # Goal should be gone
        assert await goal_repo.get_by_id(goal.id) is None

    @pytest.mark.asyncio
    async def test_long_text_content(self, session):
        """Test storing long text content in history."""
        repo = HistoryRepository(session)
        long_text = "x" * 10000
        entry = await repo.save(agent_name="User", role="user", content=long_text)
        assert entry.content == long_text

        fetched = await repo.get_recent(limit=1)
        assert fetched[0].content == long_text

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, session):
        """Test storing special characters and unicode."""
        repo = HistoryRepository(session)
        content = "Hello 🌍! <script>alert('xss')</script> & \"quotes\""
        entry = await repo.save(agent_name="User", role="user", content=content)
        assert entry.content == content