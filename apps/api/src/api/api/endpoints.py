from api.agents.retrieval_generation import rag_pipeline
from api.api.models import RAGRequest, RAGResponse
from fastapi import APIRouter, HTTPException, Request

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def chat(_request: Request, payload: RAGRequest) -> RAGResponse:
    result = rag_pipeline(payload.query)

    return RAGResponse(answer=result["answer"])


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
