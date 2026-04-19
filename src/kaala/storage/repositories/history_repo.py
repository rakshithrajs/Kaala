"""Repository for ConversationHistory CRUD operations."""

from typing import Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.storage.models import ConversationHistory


class HistoryRepository:
    """Repository for managing conversation history in the database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(
        self, agent_name: str, role: str, content: str
    ) -> ConversationHistory:
        """Save a message to conversation history."""
        entry = ConversationHistory(
            agent_name=agent_name, role=role, content=content
        )
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def get_recent(
        self, agent_name: Optional[str] = None, limit: int = 50
    ) -> list[ConversationHistory]:
        """Get recent conversation history, optionally filtered by agent."""
        query = select(ConversationHistory)
        if agent_name:
            query = query.where(ConversationHistory.agent_name == agent_name)
        result = await self.session.execute(
            query.order_by(ConversationHistory.timestamp.desc()).limit(limit)
        )
        # Return in chronological order (oldest first)
        return list(reversed(result.scalars().all()))

    async def get_by_agent(
        self, agent_name: str, limit: int = 50
    ) -> list[ConversationHistory]:
        """Get conversation history for a specific agent."""
        return await self.get_recent(agent_name=agent_name, limit=limit)

    async def clear(self, agent_name: Optional[str] = None) -> int:
        """Clear conversation history, optionally for a specific agent."""
        stmt = delete(ConversationHistory)
        if agent_name:
            stmt = stmt.where(ConversationHistory.agent_name == agent_name)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def get_context_for_agent(
        self, agent_name: str, limit: int = 10
    ) -> list[dict]:
        """Get recent context for an agent as a list of dicts.

        Returns messages in format: [{"role": "user/assistant", "content": "..."}]
        """
        entries = await self.get_by_agent(agent_name, limit=limit)
        return [{"role": e.role, "content": e.content} for e in entries]