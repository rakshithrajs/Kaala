"""Repository for Goal CRUD operations."""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models import Goal


class GoalRepository:
    """Repository for managing goals in the database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self, goal_text: str, details: Optional[str] = None, status: str = "pending"
    ) -> Goal:
        """Create a new goal."""
        goal = Goal(goal_text=goal_text, details=details, status=status)
        self.session.add(goal)
        await self.session.commit()
        await self.session.refresh(goal)
        return goal

    async def get_by_id(self, goal_id: int) -> Optional[Goal]:
        """Get a goal by ID."""
        result = await self.session.execute(select(Goal).where(Goal.id == goal_id))
        return result.scalar_one_or_none()

    async def get_all(self, status: Optional[str] = None) -> list[Goal]:
        """Get all goals, optionally filtered by status."""
        query = select(Goal)
        if status:
            query = query.where(Goal.status == status)
        result = await self.session.execute(query.order_by(Goal.created_at.desc()))
        return list(result.scalars().all())

    async def get_pending(self) -> list[Goal]:
        """Get all pending goals."""
        return await self.get_all(status="pending")

    async def update_status(
        self, goal_id: int, status: str, completed_at: Optional[datetime] = None
    ) -> Optional[Goal]:
        """Update goal status."""
        goal = await self.get_by_id(goal_id)
        if goal:
            goal.status = status
            if completed_at:
                goal.completed_at = completed_at
            elif status == "completed":
                goal.completed_at = datetime.utcnow()
            await self.session.commit()
            await self.session.refresh(goal)
        return goal

    async def update_details(self, goal_id: int, details: str) -> Optional[Goal]:
        """Update goal details."""
        goal = await self.get_by_id(goal_id)
        if goal:
            goal.details = details
            await self.session.commit()
            await self.session.refresh(goal)
        return goal

    async def delete(self, goal_id: int) -> bool:
        """Delete a goal."""
        goal = await self.get_by_id(goal_id)
        if goal:
            await self.session.delete(goal)
            await self.session.commit()
            return True
        return False