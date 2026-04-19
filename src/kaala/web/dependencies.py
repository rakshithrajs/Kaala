"""FastAPI dependency injection helpers."""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.core.orchestrator import Orchestrator
from kaala.core.scheduler import PromptScheduler
from kaala.storage.database import Database


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_scheduler(request: Request) -> PromptScheduler:
    return request.app.state.scheduler


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session from the app's shared Database instance."""
    db: Database = request.app.state.db
    async with db.async_session() as session:
        yield session