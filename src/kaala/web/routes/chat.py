"""Chat API endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from kaala.core.orchestrator import Orchestrator
from kaala.web.dependencies import get_orchestrator

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat(request: ChatRequest, orch: Orchestrator = Depends(get_orchestrator)):
    result = await orch.process_user_input(request.message)
    return result