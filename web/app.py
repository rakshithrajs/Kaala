"""FastAPI application for Kaala web UI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from config.config import get_settings
from core.orchestrator import Orchestrator
from core.scheduler import PromptScheduler
from storage.database import get_db
from web.websocket import manager
from web.routes import chat, goals, schedules, history


async def _on_prompt_result(result: dict):
    """Broadcast scheduler prompt results via WebSocket."""
    event_type = "reminder" if result.get("type") == "reminder" else "scheduled_prompt_fired"
    await manager.broadcast(event_type, result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await get_db()
    orch = Orchestrator(model=settings.default_model)
    sched = PromptScheduler(
        orchestrator=orch,
        poll_interval=settings.poll_interval,
        on_prompt_result=_on_prompt_result,
    )
    await sched.start()
    app.state.orchestrator = orch
    app.state.scheduler = sched
    yield
    await sched.stop()


app = FastAPI(title="Kaala", lifespan=lifespan)

app.include_router(chat.router, prefix="/api")
app.include_router(goals.router, prefix="/api")
app.include_router(schedules.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/", StaticFiles(directory="static", html=True), name="static")