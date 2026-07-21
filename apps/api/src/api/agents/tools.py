from langchain_core.tools import tool
from qdrant_client import QdrantClient

from api.agents.retrieval_generation import (
    process_context,
    process_retrieved_reviews,
    rerank_data,
    retrieve_data,
    retrieve_prefiltered_reviews_data,
)


@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Search available products and return the top k matching inventory items.

    Expand the customer's question into 1-5 concise search statements and issue them
    in parallel in a single turn. Each statement covers one distinct product or
    attribute; no two may express the same intent. Use natural product-description
    language. If no brand or model is specified, search broadly rather than refusing.

        "Earphones for me and a waterproof speaker"
            -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
            -> "Insulated winter jacket" | "Hiking outerwear for cold weather"

    Before calling, check what earlier calls in this conversation already returned.
    Search only for what is missing; results already retrieved remain valid and must
    not be fetched again.

    Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.

    Returns:
        A string of the top k available products, each prefixed with its ID and
        average rating.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    retrieved_context = retrieve_data(query, qdrant_client, k=20, hybrid=True)

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)
    return formatted_context


@tool
def get_formatted_reviews_context(
    query: str, parent_asins: list[str], top_k: int = 5
) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(
        query, parent_asins, qdrant_client, k=top_k
    )

    formatted_context = process_retrieved_reviews(retrieved_context)
    return formatted_context
