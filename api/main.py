from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers.computer_vision import router as computer_vision_router
from api.routers.event_data import router as event_data_router


app = FastAPI(
    title="Football Analytics API",
    version="1.0.0",
    description="Backend HTTP para el frontend React de Tactical Intelligence Platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(event_data_router)
app.include_router(computer_vision_router)

computer_vision_output_dir = Path(__file__).resolve().parents[1] / "outputs" / "api"
computer_vision_output_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/api/static/computer-vision",
    StaticFiles(directory=str(computer_vision_output_dir)),
    name="computer-vision-static",
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
