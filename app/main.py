"""FastAPI application factory.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import routes_chat, routes_feedback, routes_health, routes_ingest, routes_search
from app.config import get_settings
from app.container import Container, build_container
from app.exceptions import AppError, ConfigurationError

log = logging.getLogger("app")


def create_app(container: Optional[Container] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container is not None:
            app.state.container = container
        else:
            settings = get_settings()
            problems = settings.validate_live()
            if problems:
                raise ConfigurationError("Invalid configuration:\n - " + "\n - ".join(problems))
            app.state.container = build_container(settings)
            log.info("started in %s mode (repo=%s, embeddings=%s, llm=%s)", settings.app_mode,
                     app.state.container.repo.backend, app.state.container.embedder.name,
                     app.state.container.llm.name)
        yield
        app.state.container.repo.close()

    app = FastAPI(title="Developer Support Intelligence", version=__version__,
                  description="RAG support agent over documentation and GitHub issues.", lifespan=lifespan)

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
                                                                "message": "Unexpected server error"}})

    for r in (routes_health.router, routes_chat.router, routes_search.router,
              routes_ingest.router, routes_feedback.router):
        app.include_router(r)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
