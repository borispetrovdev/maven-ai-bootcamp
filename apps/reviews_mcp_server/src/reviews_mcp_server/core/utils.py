from typing import TypedDict

import openai
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
)

embedding_model = "text-embedding-3-small"


RetrievedItemsData = TypedDict(
    "RetrievedItemsData",
    {
        "retrieved_context_ids": list[str],
        "retrieved_context": list[str],
        "similarity_scores": list[float],
        "retrieved_context_ratings": list[float],
    },
)

HYBRID_SEARCH_COLLECTION_NAME = "Amazon-items-collection-01-hybrid-search"


def get_embedding(text, model=embedding_model):
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding


RetrievedReviewsData = TypedDict(
    "RetrievedReviewsData",
    {
        "retrieved_asins": list[str],
        "retrieved_reviews": list[str],
        "similarity_scores": list[float],
    },
)


class ReviewEmbeddingPayload(BaseModel):
    """Fields embedded and stored as the Qdrant point payload."""

    preprocessed_data: str
    parent_asin: str


def retrieve_prefiltered_reviews_data(
    query: str, parent_asins: list[str], qdrant_client: QdrantClient, k=5
) -> RetrievedReviewsData:
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin", match=MatchAny(any=parent_asins)
                        )
                    ]
                ),
                limit=20,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
    )

    retrieved_asins = []
    retrieved_reviews = []
    similarity_scores = []

    for result in results.points:
        if not result.payload:
            raise ValueError("No payload found in Qdrant ScoredPoint")
        payload = ReviewEmbeddingPayload.model_validate(result.payload)
        retrieved_asins.append(payload.parent_asin)
        retrieved_reviews.append(payload.preprocessed_data)
        similarity_scores.append(result.score)

    return {
        "retrieved_asins": retrieved_asins,
        "retrieved_reviews": retrieved_reviews,
        "similarity_scores": similarity_scores,
    }


def process_retrieved_reviews(retrieved_data: RetrievedReviewsData) -> str:
    formatted_context = ""

    for asin, review in zip(
        retrieved_data["retrieved_asins"], retrieved_data["retrieved_reviews"]
    ):
        formatted_context += f"- ID: {asin}, user review: {review}\n"

    return formatted_context
