"""Conversation history API endpoint."""

from fastapi import APIRouter, Query

from storage.database import get_db
from storage.repositories import HistoryRepository

router = APIRouter()


@router.get("/history")
async def get_history(
    agent: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    db = await get_db()
    async with db.async_session() as session:
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