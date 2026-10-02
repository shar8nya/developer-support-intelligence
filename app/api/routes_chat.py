from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_api_key
from app.container import Container
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    """Answer a question with a grounded, cited answer - or abstain when evidence is insufficient."""
    return c.rag.answer(req)
