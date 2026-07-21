import logging

from fastapi import APIRouter, Request

from api.agents.graph import agent_wrapper
from api.api.models import (
    AgentRequest,
    AgentResponse,
    FeedbackRequest,
    FeedbackResponse,
    RAGUsedContext,
)
from api.api.processors.submit_feedback import submit_feedback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/")
def chat(_request: Request, payload: AgentRequest) -> AgentResponse:
    result = agent_wrapper(payload.query, payload.thread_id)

    return AgentResponse(
        answer=result["answer"],
        used_context=[RAGUsedContext(**item) for item in result["used_context"]],
        trace_id=result["trace_id"],
    )


@feedback_router.post("/")
def send_feedback(_request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    submit_feedback(payload)
    return FeedbackResponse(message="Feedback submitted successfully")


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])
