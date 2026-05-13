from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

project_root = Path(__file__).resolve().parents[1]
computer_vision_output_dir = Path(__file__).resolve().parents[1] / "outputs" / "api"
computer_vision_output_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/api/static/computer-vision",
    StaticFiles(directory=str(computer_vision_output_dir)),
    name="computer-vision-static",
)

frontend_dist_dir = project_root / "front-tip" / "dist"
frontend_index_file = frontend_dist_dir / "index.html"


def _resolve_frontend_asset(relative_path: str) -> Path | None:
    if not relative_path or not frontend_dist_dir.exists():
        return None

    candidate = (frontend_dist_dir / relative_path).resolve()
    try:
        candidate.relative_to(frontend_dist_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Asset no encontrado.")

    if candidate.is_file():
        return candidate
    return None


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_frontend_index() -> FileResponse:
    if not frontend_index_file.exists():
        raise HTTPException(status_code=404, detail="Build del frontend no encontrado.")
    return FileResponse(frontend_index_file)


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_app(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Recurso API no encontrado.")

    asset_path = _resolve_frontend_asset(full_path)
    if asset_path:
        return FileResponse(asset_path)

    if frontend_index_file.exists():
        return FileResponse(frontend_index_file)

    raise HTTPException(status_code=404, detail="Build del frontend no encontrado.")
