import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

embedding_model = "text-embedding-3-small"

qdrant_client = QdrantClient(url="http://qdrant:6333")


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


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, k=5):
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
def process_context(context):
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
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model=answer_gen_model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning_effort="none",
    )
    current_run = get_current_run_tree()
    if not current_run:
        raise ValueError("No current run found")
    if not response.usage:
        raise ValueError("No usage metadata found in response")
    current_run.metadata["usage_metadata"] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content


@traceable(
    name="rag_pipeline",
)
def rag_pipeline(question, topk_k=5):

    retrieved_context = retrieve_data(question, k=topk_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    if not answer:
        raise ValueError("LLM returned no content")

    final_answer = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
    }

    return final_answer
