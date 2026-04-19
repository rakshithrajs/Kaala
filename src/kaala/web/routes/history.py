"""Conversation history API endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.storage.repositories import HistoryRepository
from kaala.web.dependencies import get_db_session

router = APIRouter()


@router.get("/history")
async def get_history(
    agent: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db_session),
):
    repo = HistoryRepository(session)
    entries = await repo.get_recent(agent_name=agent, limit=limit)
    return [
        {
            "agent": e.agent_name,
            "role": e.role,
            "content": e.content,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
        }
        for e in entries
    ]