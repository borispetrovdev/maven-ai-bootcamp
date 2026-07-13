from typing import Literal

import requests
import streamlit as st
from api.api.models import RAGResponse, RAGUsedContext
from pydantic import BaseModel

from chatbot_ui.core.config import config


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AppState(BaseModel, validate_assignment=True):
    messages: list[ChatMessage] = [
        ChatMessage(role="assistant", content="Hello! How can I assist you today?")
    ]
    used_context: list[RAGUsedContext] = []


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


state = get_state()

for message in state.messages:
    with st.chat_message(message.role):
        st.write(message.content)

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


if prompt := st.chat_input("Hello! How can I assist you today?"):
    state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status, output = api_call(
            "POST",
            f"{config.API_URL}/agent/",
            json={"query": prompt},
        )
        if not status:
            st.error(output.get("detail") or output.get("message") or "Request failed")
            st.stop()
        response_data = RAGResponse.model_validate(output)
        answer = response_data.answer

        state.used_context = response_data.used_context

        st.write(answer)
    state.messages.append(ChatMessage(role="assistant", content=answer))
    st.rerun()
