from api.api.models import FeedbackRequest
from langsmith import Client

client = Client()


def submit_feedback(payload: FeedbackRequest) -> None:
    if payload.feedback_score is not None:
        client.create_feedback(
            run_id=payload.trace_id,
            key="thumbs",
            value=payload.feedback_score,
            feedback_source_type=payload.feedback_source_type,
        )

    if len(payload.feedback_text) > 0:
        client.create_feedback(
            run_id=payload.trace_id,
            key="comment",
            value=payload.feedback_text,
            feedback_source_type=payload.feedback_source_type,
        )
