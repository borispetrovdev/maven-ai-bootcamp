import streamlit as st
import requests
from chatbot_ui.core.config import config


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


with st.sidebar:
    st.title("Settings")

    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])
    elif provider == "Groq":
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile"])
    elif provider == "Google":
        model_name = st.selectbox("Model", ["gemini-2.5-flash"])
    else:
        raise ValueError(f"Invalid provider: {provider}")

    st.session_state.provider = provider
    st.session_state.model_name = model_name

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = api_call(
            "POST",
            f"{config.API_URL}/chat",
            json={
                "provider": st.session_state.provider,
                "model_name": st.session_state.model_name,
                "messages": st.session_state.messages,
            },
        )
        response_data = output[1]
        answer = response_data["message"]
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
