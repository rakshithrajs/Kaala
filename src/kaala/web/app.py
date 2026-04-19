"""FastAPI application for Kaala web UI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kaala.config.settings import get_settings
from kaala.core.orchestrator import Orchestrator
from kaala.core.scheduler import PromptScheduler
from kaala.storage.database import get_db
from kaala.web.websocket import manager
from kaala.web.routes import chat, goals, schedules, history

STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"


async def _on_prompt_result(result: dict):
    """Broadcast scheduler prompt results via WebSocket."""
    event_type = "reminder" if result.get("type") == "reminder" else "scheduled_prompt_fired"
    await manager.broadcast(event_type, result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db = await get_db()
    orch = Orchestrator(model=settings.default_model)
    sched = PromptScheduler(
        orchestrator=orch,
        poll_interval=settings.poll_interval,
        on_prompt_result=_on_prompt_result,
    )
    await sched.start()
    app.state.orchestrator = orch
    app.state.scheduler = sched
    app.state.db = db
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


# SPA fallback: serve index.html for any non-API, non-WS, non-static-file route
# This must come before the static mount so client-side routing works.
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the React SPA for all non-API routes."""
    file_path = STATIC_DIR / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")