import logging

import instructor
from langchain_core.messages import (
    SystemMessage,
    convert_to_openai_messages,
)
from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree, traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from openai.types.responses.response import Response
from pydantic import BaseModel, Field

from api.agents.tools import get_formatted_item_context
from api.agents.utils.prompt_management import prompt_template_config
from api.api.models import RAGUsedContextSimple, State, StateUpdate

PROVIDER_NAME_AGENT = "openai"
MODEL_NAME_AGENT = "gpt-5.4-mini"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FinalResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContextSimple] = Field(
        description="List of items used to answer the question"
    )


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(
        description="A clarifying question, if the user's initial query is not relevant."
    )


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def agent_node(state: State) -> StateUpdate:

    template = prompt_template_config("api/agents/prompts/qna_agent.yml", "qna_agent")

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="none", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, FinalResponse], tool_choice="any"
    )

    final_answer = False
    answer = ""
    references: list[RAGUsedContextSimple] = []
    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])

    current_run = get_current_run_tree()
    if current_run and response.usage_metadata:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
        }

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == FinalResponse.__name__:
                final_answer = True
                final_response_validated = FinalResponse.model_validate(
                    tool_call.get("args")
                )
                references.extend(final_response_validated.references)
                answer = final_response_validated.answer
                break

    return {
        "messages": [response],
        "final_answer": final_answer,
        "iteration": state.iteration + 1,
        "answer": answer,
        "references": references,
    }


@traceable(
    name="route_intent",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def intent_router_node(state: State) -> IntentRouterResponse:

    template = prompt_template_config(
        "api/agents/prompts/intent_router_agent.yml", "intent_router_agent"
    )

    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(
        wrap_openai(OpenAI()),
        mode=instructor.Mode.RESPONSES_TOOLS,
    )

    response, raw_response = client.create_with_completion(
        model=MODEL_NAME_AGENT,
        messages=[{"role": "system", "content": prompt}, *conversation],  # type: ignore[arg-type]
        reasoning={"effort": "none"},
        response_model=IntentRouterResponse,
    )

    if not isinstance(raw_response, Response) or raw_response.usage is None:
        raise ValueError(f"Unexpected raw response: {type(raw_response)}")

    current_run = get_current_run_tree()
    if not current_run:
        raise ValueError("No current run found")
    if not raw_response.usage:
        raise ValueError("No usage metadata found in response")
    current_run.metadata["usage_metadata"] = {
        "input_tokens": raw_response.usage.input_tokens,
        "output_tokens": raw_response.usage.output_tokens,
        "total_tokens": raw_response.usage.total_tokens,
    }

    return response
