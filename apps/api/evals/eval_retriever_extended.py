import logging
from typing import Sequence, cast

from api.agents.retrieval_generation import RAGPipelineResponse, rag_pipeline
from langsmith import Client, EvaluationResult
from langsmith.evaluation._runner import EVALUATOR_T
from langsmith.schemas import Example, Run
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from apps.api.evals.models import RagEvalReferenceOutput

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

openai_client = AsyncOpenAI()
ragas_llm = llm_factory(
    "gpt-4.1-mini",
    client=openai_client,
    max_completion_tokens=4000,
    # temperature=1.0,
)
ragas_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", client=openai_client
)

# gpt-5.4-mini uses decimal versioning; ragas only auto-maps max_tokens for gpt-5, gpt-6, etc.
ragas_llm.model_args.pop(  # pyright: ignore[reportAttributeAccessIssue]
    "max_tokens", None
)
ragas_llm.model_args.pop("top_p", None)  # pyright: ignore[reportAttributeAccessIssue]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def context_precision_id_based(run: Run, example: Example | None) -> EvaluationResult:
    if run.outputs is None or example is None or example.outputs is None:
        return EvaluationResult(score=0, key="context_precision_id_based")
    run_outputs = cast(RAGPipelineResponse, run.outputs)
    retrieved_context_ids = {str(id) for id in run_outputs["retrieved_context_ids"]}
    example_outputs = RagEvalReferenceOutput.model_validate(example.outputs)
    try:
        reference_context_ids = {
            str(id) for id in example_outputs.reference_context_ids
        }
    except KeyError:
        logger.error(f"No reference context ids found in example: {example}")
        return EvaluationResult(score=0, key="context_precision_id_based")

    score = (
        len(retrieved_context_ids & reference_context_ids) / len(retrieved_context_ids)
        if retrieved_context_ids
        else 0
    )

    return EvaluationResult(score=score, key="context_precision_id_based")


def context_recall_id_based(run: Run, example: Example | None) -> EvaluationResult:
    if run.outputs is None or example is None or example.outputs is None:
        return EvaluationResult(score=0, key="context_recall_id_based")

    run_outputs = cast(RAGPipelineResponse, run.outputs)
    example_outputs = RagEvalReferenceOutput.model_validate(example.outputs)
    retrieved_context_ids = {str(id) for id in run_outputs["retrieved_context_ids"]}
    reference_context_ids = {str(id) for id in example_outputs.reference_context_ids}

    score = (
        len(retrieved_context_ids & reference_context_ids) / len(reference_context_ids)
        if reference_context_ids
        else 0
    )

    return EvaluationResult(score=score, key="context_recall_id_based")


def ragas_faithfulness(run: Run) -> EvaluationResult:
    scorer = Faithfulness(llm=ragas_llm)
    if run.outputs is None:
        return EvaluationResult(score=0, key="ragas_faithfulness")
    run_outputs = cast(RAGPipelineResponse, run.outputs)
    return scorer.score(
        user_input=run_outputs["question"],
        response=run_outputs["answer"],
        retrieved_contexts=run_outputs["retrieved_context"],
    ).value


def ragas_relevancy(run: Run) -> EvaluationResult:
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    if run.outputs is None:
        return EvaluationResult(score=0, key="ragas_relevancy")
    run_outputs = cast(RAGPipelineResponse, run.outputs)
    return scorer.score(
        user_input=run_outputs["question"], response=run_outputs["answer"]
    ).value


evaluators: Sequence[EVALUATOR_T] = [
    context_precision_id_based,
    context_recall_id_based,
    ragas_faithfulness,
    ragas_relevancy,
]

print("Evaluating plain retriever")

result = ls_client.evaluate(
    lambda x: dict(
        rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=False, rerank=False)
    ),
    data="rag-evaluation-dataset-extended",
    evaluators=evaluators,
    experiment_prefix="plain",
)

print("Evaluating hybrid retriever")

result = ls_client.evaluate(
    lambda x: dict(
        rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=True, rerank=False)
    ),
    data="rag-evaluation-dataset-extended",
    evaluators=evaluators,
    experiment_prefix="hybrid",
)

print("Evaluating hybrid retriever with rerank")

result = ls_client.evaluate(
    lambda x: dict(
        rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=True, rerank=True)
    ),
    data="rag-evaluation-dataset-extended",
    evaluators=evaluators,
    experiment_prefix="hybrid-rerank",
    max_concurrency=10,
)
