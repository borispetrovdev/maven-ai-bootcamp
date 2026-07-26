from typing import TypedDict

import cohere
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Document, Prefetch, Rrf, RrfQuery

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


def retrieve_items_data(
    query, qdrant_client: QdrantClient, k=5, hybrid=True
) -> RetrievedItemsData:
    query_embedding = get_embedding(query)
    if hybrid:
        results = qdrant_client.query_points(
            collection_name=HYBRID_SEARCH_COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="text-embedding-3-small",
                    limit=20,
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="qdrant/bm25",
                    ),
                    using="bm25",
                    limit=20,
                ),
            ],
            query=RrfQuery(rrf=Rrf(weights=[3, 1])),
            limit=k,
        )
    else:
        results = qdrant_client.query_points(
            collection_name=HYBRID_SEARCH_COLLECTION_NAME,
            query=query_embedding,
            using="text-embedding-3-small",
            limit=k,
        )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        if not result.payload:
            raise ValueError("No payload found in Qdrant ScoredPoint")
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["preprocessed_description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings,
    }


RERANK_MODEL = "rerank-v4.0-pro"


def rerank_data(query: str, context: RetrievedItemsData, top_k=5) -> RetrievedItemsData:
    cohere_client = cohere.ClientV2()
    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=context["retrieved_context"],
        top_n=top_k,
    )
    order = [result.index for result in response.results]
    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [
            context["retrieved_context_ratings"][i] for i in order
        ],
    }


def process_context(context: RetrievedItemsData) -> str:
    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context
