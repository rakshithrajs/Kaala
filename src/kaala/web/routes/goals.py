"""Goals CRUD API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from kaala.storage.repositories import GoalRepository
from kaala.web.dependencies import get_db_session

router = APIRouter()


class GoalUpdate(BaseModel):
    status: str | None = None
    details: str | None = None


def _goal_to_dict(g) -> dict:
    return {
        "id": g.id,
        "goal": g.goal_text,
        "details": g.details,
        "status": g.status,
        "created_at": g.created_at.isoformat() if g.created_at else None,
        "completed_at": g.completed_at.isoformat() if g.completed_at else None,
    }


@router.get("/goals")
async def list_goals(status: str | None = Query(None), session: AsyncSession = Depends(get_db_session)):
    repo = GoalRepository(session)
    goals = await repo.get_all(status=status)
    return [_goal_to_dict(g) for g in goals]


@router.get("/goals/{goal_id}")
async def get_goal(goal_id: int, session: AsyncSession = Depends(get_db_session)):
    repo = GoalRepository(session)
    goal = await repo.get_by_id(goal_id)
    if not goal:
        raise HTTPException(404, "Goal not found")
    return _goal_to_dict(goal)


@router.patch("/goals/{goal_id}")
async def update_goal(goal_id: int, update: GoalUpdate, session: AsyncSession = Depends(get_db_session)):
    repo = GoalRepository(session)
    goal = None
    if update.status is not None:
        goal = await repo.update_status(goal_id, update.status)
    if update.details is not None:
        goal = await repo.update_details(goal_id, update.details)
    if not goal:
        raise HTTPException(404, "Goal not found")
    return _goal_to_dict(goal)


@router.delete("/goals/{goal_id}")
async def delete_goal(goal_id: int, session: AsyncSession = Depends(get_db_session)):
    repo = GoalRepository(session)
    deleted = await repo.delete(goal_id)
    if not deleted:
        raise HTTPException(404, "Goal not found")
    return {"ok": True}