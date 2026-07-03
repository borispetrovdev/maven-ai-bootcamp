import logging
from typing import TypedDict

import instructor
import openai
from langsmith import get_current_run_tree, traceable
from openai.types.responses.response import Response
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.api.models import ItemPayload

embedding_model = "text-embedding-3-small"


class RAGUsedContextSimple(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(
        description="Description of the item corresponding to the id"
    )


class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContextSimple] = Field(
        description="List of items used to answer the question"
    )


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": embedding_model},
)
def get_embedding(text, model=embedding_model):
    response = openai.embeddings.create(input=text, model=model)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding


RetrievedData = TypedDict(
    "RetrievedData",
    {
        "retrieved_context_ids": list[str],
        "retrieved_context": list[str],
        "similarity_scores": list[float],
        "retrieved_context_ratings": list[float],
    },
)


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, qdrant_client: QdrantClient, k=5) -> RetrievedData:
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01", query=query_embedding, limit=k
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


@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context: RetrievedData) -> str:
    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- Answer the question based on the provided context only.
- Never use word context and refer to it as the available products.
- Do not use markdown formatting.

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt


answer_gen_model = "gpt-5.4-nano"


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": answer_gen_model},
)
def generate_answer(prompt: str) -> RAGGenerationResponse:

    client = instructor.from_provider(
        "openai/" + answer_gen_model, mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning={"effort": "none"},
        response_model=RAGGenerationResponse,
    )

    if not isinstance(raw_response, Response) or raw_response.usage is None:
        raise ValueError(f"Unexpected raw response: {type(raw_response)}")

    current_run = get_current_run_tree()
    if not current_run:
        raise ValueError("No current run found")
    if not raw_response.usage:
        raise ValueError("No usage metadata found in response")
    current_run.metadata["usage_metadata"] = {
        "input_tokens": raw_response.usage.input_tokens,
        "output_tokens": raw_response.usage.output_tokens,
        "total_tokens": raw_response.usage.total_tokens,
    }

    return response


RAGPipelineResponse = TypedDict(
    "RAGPipelineResponse",
    {
        "answer": str,
        "references": list[RAGUsedContextSimple],
        "question": str,
        "retrieved_context_ids": list[str],
        "retrieved_context": list[str],
    },
)


@traceable(
    name="rag_pipeline",
)
def rag_pipeline(
    question: str, qdrant_client: QdrantClient, topk_k=5
) -> RAGPipelineResponse:

    retrieved_context = retrieve_data(question, qdrant_client, k=topk_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    if not answer:
        raise ValueError("LLM returned no content")

    final_answer: RAGPipelineResponse = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
    }

    return final_answer


UsedContextEntry = TypedDict(
    "UsedContextEntry",
    {"id": str, "image_url": str, "price": float | None, "description": str},
)

RAGPipelineWithDecorationResponse = TypedDict(
    "RAGPipelineWithDecorationResponse",
    {
        "answer": str,
        "used_context": list[UsedContextEntry],
    },
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def rag_pipeline_with_decoration(
    question, topk_k=5
) -> RAGPipelineWithDecorationResponse:
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = rag_pipeline(question, qdrant_client, topk_k)

    used_context: list[UsedContextEntry] = []
    for item in result.get("references", []):
        points = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
                ]
            ),
        )[0]

        payload = points[0].payload
        if not payload:
            raise ValueError(f"No payload in point: {points[0].id}")
        payload = ItemPayload.model_validate(payload)
        used_context.append(
            {
                "id": item.id,
                "image_url": str(payload.image),
                "price": payload.price,
                "description": payload.preprocessed_description,
            }
        )

    return {
        "answer": result["answer"],
        "used_context": used_context,
    }
