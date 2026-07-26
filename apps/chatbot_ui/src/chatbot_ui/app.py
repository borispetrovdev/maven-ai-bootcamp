import json
import uuid
from collections.abc import Iterator
from typing import Literal, TypedDict, assert_never

import requests
import streamlit as st
from api.agents.graph import AgentStreamResponse
from api.api.models import (
    AgentRequest,
    AgentResponse,
    FeedbackRequest,
    FeedbackResponse,
    RAGUsedContext,
)
from pydantic import BaseModel, ValidationError

from chatbot_ui.core.config import config


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AppState(BaseModel, validate_assignment=True):
    messages: list[ChatMessage] = [
        ChatMessage(role="assistant", content="Hello! How can I assist you today?")
    ]
    used_context: list[RAGUsedContext] = []
    thread_id: str = str(uuid.uuid4())
    trace_id: str = ""
    latest_feedback: Literal["positive", "negative"] | None = None
    feedback_submission_status: Literal["success", "error"] | None = None
    show_feedback_box: bool = False


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Shopping Assistant")
st.write("Ask me anything about the products in stock.")


def api_call(method: str, url: str, **kwargs):
    def _show_error_popup(message: str) -> None:
        st.session_state["error_popup"] = {"message": message, "visible": True}

    try:
        response = getattr(requests, method.lower())(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup(
            "Connection error. Please check your internet connection and try again."
        )
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An error occurred: {str(e)}")
        return False, {"message": str(e)}


def api_call_stream(method: str, url: str, **kwargs) -> Iterator[bytes]:
    """Yield the response's lines, or nothing at all once an error popup is shown."""

    def _show_error_popup(message: str) -> None:
        st.session_state["error_popup"] = {"message": message, "visible": True}

    try:
        response: requests.Response = getattr(requests, method.lower())(url, **kwargs)

        return response.iter_lines()

    except requests.exceptions.ConnectionError:
        _show_error_popup(
            "Connection error. Please check your internet connection and try again."
        )
        return iter(())
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return iter(())
    except Exception as e:
        _show_error_popup(f"An error occurred: {str(e)}")
        return iter(())


state = get_state()


class SuccessResult(TypedDict):
    success: Literal[True]
    response: FeedbackResponse


class FailureResult(TypedDict):
    success: Literal[False]
    response: None


FeedbackRequestResponse = SuccessResult | FailureResult


def submit_feedback(
    feedback_type: Literal["positive", "negative"],
    feedback_text: str,
) -> FeedbackRequestResponse:
    """Submit feedback to the API"""

    def _feedback_score(feedback_type: Literal["positive", "negative"]) -> int:
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            assert_never(feedback_type)

    feedback_data = FeedbackRequest(
        trace_id=state.trace_id,
        feedback_score=_feedback_score(feedback_type),
        feedback_text=feedback_text,
        feedback_source_type="api",
    )

    status, response = api_call(
        "POST",
        f"{config.API_URL}/submit_feedback",
        json=feedback_data.model_dump(),
    )
    if status:
        return SuccessResult(
            success=True, response=FeedbackResponse.model_validate(response)
        )
    else:
        return FailureResult(success=False, response=None)


with st.sidebar:
    if len(state.used_context) > 0:
        (suggestions_tab,) = st.tabs(["🔍 Suggestions"])

        with suggestions_tab:
            for idx, item in enumerate(state.used_context):
                st.caption(item.description)
                st.image(item.image_url, width=250)
                if (price := item.price) is not None:
                    st.write(f"Price: ${price} USD")
                st.divider()

for idx, message in enumerate(state.messages):
    with st.chat_message(message.role):
        st.markdown(message.content)

        # Add feedback buttons only for the latest assistant message (excluding the initial greeting)
        is_latest_assistant = (
            message.role == "assistant" and idx == len(state.messages) - 1 and idx > 0
        )

        if is_latest_assistant:
            # Use Streamlit's built-in feedback component
            feedback_key = f"feedback_{len(state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)

            # Handle feedback selection
            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"

                # Only submit if this is a new/different feedback
                if state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        result = submit_feedback(
                            feedback_type=feedback_type, feedback_text=""
                        )
                        if result["success"]:
                            state.latest_feedback = feedback_type
                            state.feedback_submission_status = "success"
                            state.show_feedback_box = feedback_type == "negative"
                        else:
                            state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()

            # Show feedback status message
            if state.latest_feedback and state.feedback_submission_status == "success":
                if state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif (
                    state.latest_feedback == "negative" and not state.show_feedback_box
                ):
                    st.success("✅ Thank you for your feedback!")
            elif state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")

            # Show feedback text box if thumbs down was pressed
            if state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption(
                    "Your negative feedback has already been recorded. You can optionally provide additional details below."
                )

                # Text area for detailed feedback
                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{len(state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100,
                )

                # Send additional feedback button
                col_send, col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button(
                        "Send Additional Details",
                        key=f"send_additional_{len(state.messages)}",
                    ):
                        if feedback_text.strip():  # Only send if there's actual text
                            with st.spinner("Submitting additional feedback..."):
                                result = submit_feedback(
                                    feedback_type="negative",
                                    feedback_text=feedback_text,
                                )
                                if result["success"]:
                                    st.success(
                                        "✅ Thank you! Your additional feedback has been recorded."
                                    )
                                    state.show_feedback_box = False
                                else:
                                    st.error(
                                        "❌ Failed to submit additional feedback. Please try again."
                                    )
                        else:
                            st.warning(
                                "Please enter some feedback text before submitting."
                            )
                        st.rerun()

                with col_close:
                    if st.button("Close", key=f"close_feedback_{len(state.messages)}"):
                        state.show_feedback_box = False


if prompt := st.chat_input("Hello! How can I assist you today?"):
    state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        message_placeholder = st.empty()
        for line in api_call_stream(
            "POST",
            f"{config.API_URL}/agent/",
            json=AgentRequest(query=prompt, thread_id=state.thread_id).model_dump(),
            stream=True,
            headers={"Accept": "text/event-stream"},
        ):
            line_text: str = line.decode("utf-8")
            if line_text.startswith("data: "):
                data = line_text[6:]

                try:
                    json_data = json.loads(data)
                    try:
                        response = AgentStreamResponse.model_validate(json_data)
                        data = AgentResponse.model_validate(response.data)
                        state.used_context = data.used_context
                        state.messages.append(
                            ChatMessage(role="assistant", content=data.answer)
                        )
                        state.trace_id = data.trace_id

                        state.latest_feedback = None
                        state.show_feedback_box = False
                        state.feedback_submission_status = None

                        status_placeholder.empty()
                        message_placeholder.markdown(data.answer)
                    except ValidationError as e:
                        st.error(
                            f"Response was not a valid AgentStreamResponse or data was not a valid AgentResponse: {e}. Response: {json_data}"
                        )

                except json.JSONDecodeError:
                    status_placeholder.markdown(f"*{data}*")

    st.rerun()
