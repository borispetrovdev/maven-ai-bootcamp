from fastapi import FastAPI, Request
from pydantic import BaseModel

from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_llm(
    provider: str, model_name: str, messages: list[dict], max_tokens: int = 500
) -> str | None:
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return (
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                reasoning_effort="minimal",
            )
            .choices[0]
            .message.content
        )
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
        return (
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )
    elif provider == "Google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=model_name, contents=[message["content"] for message in messages]
        )
        return response.text
    else:
        raise ValueError(f"Invalid provider: {provider}")


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]


class ChatResponse(BaseModel):
    message: str


app = FastAPI()


@app.post("/chat")
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    result = run_llm(
        payload.provider,
        payload.model_name,
        payload.messages,
    )

    return ChatResponse(message=result)
