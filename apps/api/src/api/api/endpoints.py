import logging

from fastapi import APIRouter, Request

from api.agents.graph import agent_wrapper
from api.api.models import AgentRequest, AgentResponse, RAGUsedContext

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def chat(_request: Request, payload: AgentRequest) -> AgentResponse:
    result = agent_wrapper(payload.query, payload.thread_id)

    return AgentResponse(
        answer=result["answer"],
        used_context=[RAGUsedContext(**item) for item in result["used_context"]],
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])
