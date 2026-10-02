from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_container
from app.container import Container
from app.exceptions import AppError
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(c: Container = Depends(get_container)) -> HealthResponse:
    problems = c.settings.validate_live()
    db_state, docs, chunks = "ok", 0, 0
    try:
        c.repo.ping()
        st = c.repo.stats()
        docs, chunks = st["documents"], st["chunks"]
    except AppError as exc:
        db_state = "unreachable"
        problems.append(exc.message)
    return HealthResponse(
        status="ok" if not problems else "degraded",
        mode="demo" if c.settings.is_demo else "live", version=__version__,
        database=f"{c.repo.backend}:{db_state}", embedding_provider=c.embedder.name,
        llm_provider=f"{c.llm.name}:{c.llm.model}", documents=docs, chunks=chunks, problems=problems,
    )
