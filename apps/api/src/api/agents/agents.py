from google import genai
from groq import Groq
from openai import OpenAI

from api.core.config import config


def run_llm(
    provider: str, model_name: str, messages: list[dict], max_tokens: int = 500
) -> str | None:
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return (
            client.chat.completions.create(
                model=model_name,
                # Plain dicts are the right shape but don't match the SDK's
                # TypedDict union; deliberately skipping precise typing here.
                messages=messages,  # pyright: ignore[reportArgumentType]
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
                messages=messages,  # pyright: ignore[reportArgumentType]
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
