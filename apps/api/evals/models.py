from pydantic import BaseModel


class RagEvalReferenceOutput(BaseModel):
    reference_context_ids: list[str]
    reference_descriptions: list[str]
    ground_truth: str
