"""Repository for UserContext CRUD operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.storage.models import UserContext


class UserContextRepository:
    """Repository for managing user context in the database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def set(self, key: str, value: str) -> UserContext:
        """Set or update a context value."""
        result = await self.session.execute(
            select(UserContext).where(UserContext.key == key)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = value
        else:
            existing = UserContext(key=key, value=value)
            self.session.add(existing)
        await self.session.commit()
        await self.session.refresh(existing)
        return existing

    async def get(self, key: str) -> str | None:
        """Get a context value by key."""
        result = await self.session.execute(
            select(UserContext).where(UserContext.key == key)
        )
        entry = result.scalar_one_or_none()
        return entry.value if entry else None

    async def get_all(self) -> dict[str, str]:
        """Get all context as a dict."""
        result = await self.session.execute(select(UserContext))
        entries = list(result.scalars().all())
        return {e.key: e.value for e in entries}

    async def delete(self, key: str) -> bool:
        """Delete a context key."""
        result = await self.session.execute(
            select(UserContext).where(UserContext.key == key)
        )
        entry = result.scalar_one_or_none()
        if entry:
            await self.session.delete(entry)
            await self.session.commit()
            return True
        return False