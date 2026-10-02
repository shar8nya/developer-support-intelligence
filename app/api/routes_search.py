import time

from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_api_key
from app.container import Container
from app.schemas import SearchRequest, SearchResponse
from app.services.rag import to_search_result

router = APIRouter(tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, c: Container = Depends(get_container)) -> SearchResponse:
    """Retrieve the most relevant chunks without calling the LLM."""
    t0 = time.perf_counter()
    hits = c.retrieval.search(req.query, req.top_k, req.use_hybrid, req.use_rerank, req.source_types)
    return SearchResponse(query=req.query, results=[to_search_result(h) for h in hits],
                          latency_ms=int((time.perf_counter() - t0) * 1000))
