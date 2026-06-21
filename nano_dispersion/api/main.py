"""FastAPI 微流控纳米颗粒扩散分析服务"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ..services import initialize_services, get_task_manager

    initialize_services()
    yield
    task_manager = get_task_manager()
    task_manager.shutdown(wait=True)


app = FastAPI(
    title="微流控纳米颗粒扩散分析服务",
    description="微流控芯片中纳米颗粒扩散实验的离线分析后端API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
