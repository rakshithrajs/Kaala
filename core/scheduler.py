"""Background scheduler that executes due prompts proactively."""

from collections.abc import Awaitable
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.orchestrator import Orchestrator
from storage.database import get_db
from storage.repositories import ScheduleRepository, UserContextRepository


class PromptScheduler:
    """Periodically checks for due scheduled prompts and executes them."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        poll_interval: int = 60,
        on_prompt_result: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ):
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval
        self.on_prompt_result = on_prompt_result
        self._scheduler = AsyncIOScheduler()
        self._db = None

    async def _get_db(self):
        if self._db is None:
            self._db = await get_db()
        return self._db

    async def start(self):
        """Start the background scheduler."""
        self._scheduler.add_job(
            self._process_due_prompts,
            "interval",
            seconds=self.poll_interval,
            id="prompt_scheduler",
            name="Process due scheduled prompts",
        )
        self._scheduler.start()

    async def stop(self):
        """Stop the background scheduler."""
        self._scheduler.shutdown(wait=False)

    async def _should_suppress(self, prompt_type: str) -> bool:
        """Check if a prompt should be suppressed based on user engagement.

        If the user has ignored several check-ins in a row, suppress future ones
        to reduce annoyance.
        """
        if prompt_type != "check_in":
            return False

        db = await self._get_db()
        async with db.async_session() as session:
            ctx_repo = UserContextRepository(session)
            ignored = await ctx_repo.get("ignored_check_ins")
            if ignored and int(ignored) >= 3:
                return True
        return False

    async def _record_engagement(self, prompt_type: str, responded: bool):
        """Record whether the user engaged with a proactive prompt."""
        db = await self._get_db()
        async with db.async_session() as session:
            ctx_repo = UserContextRepository(session)
            if prompt_type == "check_in":
                key = "ignored_check_ins"
                current = await ctx_repo.get(key)
                count = int(current or "0")
                if responded:
                    count = 0
                else:
                    count += 1
                await ctx_repo.set(key, str(count))

    async def _process_due_prompts(self):
        """Query for due prompts and execute each one."""
        try:
            db = await self._get_db()
            async with db.async_session() as session:
                repo = ScheduleRepository(session)
                due = await repo.get_due()

            for prompt in due:
                if await self._should_suppress(prompt.prompt_type):
                    db = await self._get_db()
                    async with db.async_session() as session:
                        repo = ScheduleRepository(session)
                        await repo.cancel(prompt.id)
                    continue

                await self._execute_prompt(
                    prompt.id, prompt.prompt_text, prompt.prompt_type
                )
        except Exception as e:
            print(f"\n[Scheduler Error] {e}")

    async def _execute_prompt(
        self, prompt_id: int, prompt_text: str, prompt_type: str = "check_in"
    ):
        """Execute a single scheduled prompt based on its type.

        - "check_in": Route through the full agent pipeline (conversational).
        - "reminder": Display directly to the user.
        - "follow_up": Route through the full pipeline for a progress check.
        """
        try:
            if prompt_type == "reminder":
                result = {"type": "reminder", "message": prompt_text}
            else:
                result = await self.orchestrator.process_user_input(prompt_text)

            db = await self._get_db()
            async with db.async_session() as session:
                repo = ScheduleRepository(session)
                await repo.mark_executed(prompt_id)

            if self.on_prompt_result:
                await self.on_prompt_result(result)
            else:
                print(f"\n[Kaala] {prompt_text}")
                from main import print_result
                print_result(result)

            await self._record_engagement(prompt_type, responded=True)
        except Exception as e:
            db = await self._get_db()
            async with db.async_session() as session:
                repo = ScheduleRepository(session)
                await repo.mark_failed(prompt_id)
            print(f"\n[Scheduler Error] Prompt {prompt_id} failed: {e}")
            await self._record_engagement(prompt_type, responded=False)