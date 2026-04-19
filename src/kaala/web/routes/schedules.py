"""Scheduled prompts API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.storage.repositories import ScheduleRepository
from kaala.web.dependencies import get_db_session

router = APIRouter()


def _prompt_to_dict(p) -> dict:
    return {
        "id": p.id,
        "prompt": p.prompt_text,
        "scheduled_for": p.scheduled_for.isoformat() if p.scheduled_for else None,
        "prompt_type": p.prompt_type,
        "goal_id": p.goal_id,
        "status": p.status,
        "executed_at": p.executed_at.isoformat() if p.executed_at else None,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


@router.get("/schedules")
async def list_schedules(status: str | None = Query(None), session: AsyncSession = Depends(get_db_session)):
    repo = ScheduleRepository(session)
    prompts = await repo.get_all(status=status)
    return [_prompt_to_dict(p) for p in prompts]


@router.post("/schedules/{prompt_id}/cancel")
async def cancel_schedule(prompt_id: int, session: AsyncSession = Depends(get_db_session)):
    repo = ScheduleRepository(session)
    prompt = await repo.cancel(prompt_id)
    if not prompt:
        raise HTTPException(404, "Scheduled prompt not found")
    return _prompt_to_dict(prompt)