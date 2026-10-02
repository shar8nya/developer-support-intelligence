from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from app.api.deps import get_container, require_api_key
from app.container import Container
from app.exceptions import NotFoundError
from app.ingestion.pipeline import build_connector
from app.schemas import IngestJob, IngestRequest, JobStatus

router = APIRouter(tags=["ingestion"], dependencies=[Depends(require_api_key)])


class IngestAccepted(BaseModel):
    job_id: UUID
    status: JobStatus


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
def start_ingest(req: IngestRequest, background: BackgroundTasks,
                 c: Container = Depends(get_container)) -> IngestAccepted:
    """Start an ingestion job in the background; poll GET /ingest/{job_id} for progress."""
    build_connector(req, c.settings)  # fail fast (400) on bad paths / config before queuing
    job = c.repo.create_job(req.source, req.model_dump(mode="json", exclude_none=True))
    background.add_task(c.pipeline.run_job, job.id, req)
    return IngestAccepted(job_id=job.id, status=job.status)


@router.get("/ingest/{job_id}", response_model=IngestJob)
def ingest_status(job_id: UUID, c: Container = Depends(get_container)) -> IngestJob:
    job = c.repo.get_job(job_id)
    if job is None:
        raise NotFoundError(f"Ingestion job {job_id} not found")
    return job
