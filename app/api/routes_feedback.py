from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_api_key
from app.container import Container
from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"], dependencies=[Depends(require_api_key)])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(req: FeedbackRequest, c: Container = Depends(get_container)) -> FeedbackResponse:
    fid = c.repo.save_feedback(req.interaction_id, req.rating, req.comment)
    return FeedbackResponse(id=fid)
