from api.agents.retrieval_generation import rag_pipeline
from langsmith import Client
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from ragas import SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
)
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

client = AsyncOpenAI()
ragas_llm = llm_factory(
    "gpt-5.4-mini",
    client=client,
    max_completion_tokens=4096,
    temperature=1.0,
)
ragas_embeddings = RagasOpenAIEmbeddings(model="text-embedding-3-small", client=client)

# gpt-5.4-mini uses decimal versioning; ragas only auto-maps max_tokens for gpt-5, gpt-6, etc.
ragas_llm.model_args.pop(  # pyright: ignore[reportAttributeAccessIssue]
    "max_tokens", None
)
ragas_llm.model_args.pop("top_p", None)  # pyright: ignore[reportAttributeAccessIssue]


def ragas_context_precision_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )

    scorer = IDBasedContextPrecision()

    return scorer.single_turn_score(sample)


def ragas_context_recall_id_based(run, example):
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs["retrieved_context_ids"],
        reference_context_ids=example.outputs["reference_context_ids"],
    )

    scorer = IDBasedContextRecall()
    return scorer.single_turn_score(sample)


def ragas_faithfulness(run):
    scorer = Faithfulness(llm=ragas_llm)
    return scorer.score(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"],
    ).value


def ragas_relevancy(run):
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return scorer.score(
        user_input=run.outputs["question"], response=run.outputs["answer"]
    ).value


result = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_relevancy,
    ],
    experiment_prefix="retriever",
)
