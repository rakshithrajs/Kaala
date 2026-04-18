"""Repository for ScheduledPrompt CRUD operations."""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import ScheduledPrompt


class ScheduleRepository:
    """Repository for managing scheduled prompts in the database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        prompt_text: str,
        scheduled_for: datetime,
        goal_id: Optional[int] = None,
        prompt_type: str = "check_in",
        status: str = "pending",
    ) -> ScheduledPrompt:
        """Create a new scheduled prompt."""
        prompt = ScheduledPrompt(
            prompt_text=prompt_text,
            scheduled_for=scheduled_for,
            goal_id=goal_id,
            prompt_type=prompt_type,
            status=status,
        )
        self.session.add(prompt)
        await self.session.commit()
        await self.session.refresh(prompt)
        return prompt

    async def get_by_id(self, prompt_id: int) -> Optional[ScheduledPrompt]:
        """Get a scheduled prompt by ID."""
        result = await self.session.execute(
            select(ScheduledPrompt).where(ScheduledPrompt.id == prompt_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, status: Optional[str] = None) -> list[ScheduledPrompt]:
        """Get all scheduled prompts, optionally filtered by status."""
        query = select(ScheduledPrompt)
        if status:
            query = query.where(ScheduledPrompt.status == status)
        result = await self.session.execute(
            query.order_by(ScheduledPrompt.scheduled_for.asc())
        )
        return list(result.scalars().all())

    async def get_pending(self) -> list[ScheduledPrompt]:
        """Get all pending scheduled prompts."""
        return await self.get_all(status="pending")

    async def get_due(self, before: Optional[datetime] = None) -> list[ScheduledPrompt]:
        """Get all prompts that are due for execution."""
        if before is None:
            before = datetime.utcnow()
        result = await self.session.execute(
            select(ScheduledPrompt)
            .where(ScheduledPrompt.status == "pending")
            .where(ScheduledPrompt.scheduled_for <= before)
            .order_by(ScheduledPrompt.scheduled_for.asc())
        )
        return list(result.scalars().all())

    async def get_by_goal(self, goal_id: int) -> list[ScheduledPrompt]:
        """Get all scheduled prompts for a goal."""
        result = await self.session.execute(
            select(ScheduledPrompt)
            .where(ScheduledPrompt.goal_id == goal_id)
            .order_by(ScheduledPrompt.scheduled_for.asc())
        )
        return list(result.scalars().all())

    async def mark_executed(
        self, prompt_id: int, executed_at: Optional[datetime] = None
    ) -> Optional[ScheduledPrompt]:
        """Mark a prompt as executed."""
        prompt = await self.get_by_id(prompt_id)
        if prompt:
            prompt.status = "executed"
            prompt.executed_at = executed_at or datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(prompt)
        return prompt

    async def mark_failed(self, prompt_id: int) -> Optional[ScheduledPrompt]:
        """Mark a prompt as failed."""
        prompt = await self.get_by_id(prompt_id)
        if prompt:
            prompt.status = "failed"
            prompt.executed_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(prompt)
        return prompt

    async def cancel(self, prompt_id: int) -> Optional[ScheduledPrompt]:
        """Cancel a scheduled prompt."""
        prompt = await self.get_by_id(prompt_id)
        if prompt:
            prompt.status = "cancelled"
            await self.session.commit()
            await self.session.refresh(prompt)
        return prompt

    async def delete(self, prompt_id: int) -> bool:
        """Delete a scheduled prompt."""
        prompt = await self.get_by_id(prompt_id)
        if prompt:
            await self.session.delete(prompt)
            await self.session.commit()
            return True
        return False