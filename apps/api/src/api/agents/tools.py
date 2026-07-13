from langchain_core.tools import tool
from qdrant_client import QdrantClient

from api.agents.retrieval_generation import process_context, rerank_data, retrieve_data


@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")
    retrieved_context = retrieve_data(query, qdrant_client, k=20, hybrid=True)

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)
    return formatted_context
