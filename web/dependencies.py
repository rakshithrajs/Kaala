"""FastAPI dependency injection helpers."""

from fastapi import Request

from core.orchestrator import Orchestrator
from core.scheduler import PromptScheduler


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_scheduler(request: Request) -> PromptScheduler:
    return request.app.state.scheduler