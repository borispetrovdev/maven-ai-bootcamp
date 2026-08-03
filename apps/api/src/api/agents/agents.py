import logging
from typing import List, TypedDict
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field

from api.agents.nodes import AgentNode
from api.agents.tools import (
    add_to_shopping_cart,
    check_warehouse_availability,
    get_formatted_item_context,
    get_formatted_reviews_context,
    get_shopping_cart,
    remove_from_cart,
    reserve_warehouse_items,
)
from api.agents.utils.prompt_management import prompt_template_config
from api.api.models import (
    AgentProperties,
    CoordinatorAgentProperties,
    Delegation,
    FinalAgentResponse,
    FinalQnAAgentResponse,
    Plan,
    RAGUsedContextSimple,
    State,
    StateUpdate,
)

PROVIDER_NAME_AGENT = "openai"
MODEL_NAME_AGENT = "gpt-5.4-mini"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FinalResponse(BaseModel):
    """Call this tool when the final answer is possible using available context."""

    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContextSimple] = Field(
        description="List of items used to answer the question"
    )


@traceable(
    name="product_qna_agent",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def product_qna_agent(state: State) -> StateUpdate:

    template = prompt_template_config(
        "api/agents/prompts/product_qna_agent.yml", "product_qna_agent"
    )

    prompt = template.render(
        user_id=state.user_id,
        cart_id=state.cart_id,
    )

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="low", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [
            get_formatted_item_context,
            get_formatted_reviews_context,
            FinalQnAAgentResponse,
        ],
        tool_choice="required",
    )

    final_answer = False
    answer = ""
    references: list[RAGUsedContextSimple] = []
    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])

    update_current_run_metadata(response)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == FinalQnAAgentResponse.__name__:
                final_answer = True
                final_response_validated = FinalQnAAgentResponse.model_validate(
                    tool_call.get("args")
                )
                references.extend(final_response_validated.references)
                answer = final_response_validated.answer
                # FinalResponse is a structured-output "tool" that never receives a
                # ToolMessage, so returning `response` as-is would persist an
                # unanswered function call. On the next turn the checkpointer
                # replays it and the Responses API rejects the request with
                # "No tool output found for function call ...". Store a plain text
                # AIMessage instead so the conversation history stays valid.
                response = AIMessage(content=answer)
                break

    return {
        "messages": [response],
        "product_qna_agent": AgentProperties(
            final_answer=final_answer, iteration=state.product_qna_agent.iteration + 1
        ),
        "answer": answer,
        "references": references,
    }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def shopping_cart_agent(state: State) -> StateUpdate:

    template = prompt_template_config(
        "api/agents/prompts/shopping_cart_agent.yml", "shopping_cart_agent"
    )

    prompt = template.render(
        user_id=state.user_id,
        cart_id=state.cart_id,
    )

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="low", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [
            get_shopping_cart,
            add_to_shopping_cart,
            remove_from_cart,
            FinalAgentResponse,
        ],
        tool_choice="required",
    )

    final_answer = False
    answer = ""
    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])

    update_current_run_metadata(response)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == FinalAgentResponse.__name__:
                final_answer = True
                final_response_validated = FinalAgentResponse.model_validate(
                    tool_call.get("args")
                )
                answer = final_response_validated.answer
                # FinalResponse is a structured-output "tool" that never receives a
                # ToolMessage, so returning `response` as-is would persist an
                # unanswered function call. On the next turn the checkpointer
                # replays it and the Responses API rejects the request with
                # "No tool output found for function call ...". Store a plain text
                # AIMessage instead so the conversation history stays valid.
                response = AIMessage(content=answer)
                break

    return {
        "messages": [response],
        "shopping_cart_agent": AgentProperties(
            final_answer=final_answer, iteration=state.shopping_cart_agent.iteration + 1
        ),
        "answer": answer,
    }


@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def warehouse_manager_agent(state: State) -> StateUpdate:

    template = prompt_template_config(
        "api/agents/prompts/warehouse_manager_agent.yml", "warehouse_manager_agent"
    )

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="low", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [
            check_warehouse_availability,
            reserve_warehouse_items,
            FinalAgentResponse,
        ],
        tool_choice="required",
    )

    final_answer = False
    answer = ""
    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])

    update_current_run_metadata(response)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == FinalAgentResponse.__name__:
                final_answer = True
                final_response_validated = FinalAgentResponse.model_validate(
                    tool_call.get("args")
                )
                answer = final_response_validated.answer
                # FinalResponse is a structured-output "tool" that never receives a
                # ToolMessage, so returning `response` as-is would persist an
                # unanswered function call. On the next turn the checkpointer
                # replays it and the Responses API rejects the request with
                # "No tool output found for function call ...". Store a plain text
                # AIMessage instead so the conversation history stays valid.
                response = AIMessage(content=answer)
                break

    return {
        "messages": [response],
        "warehouse_manager_agent": AgentProperties(
            final_answer=final_answer,
            iteration=state.warehouse_manager_agent.iteration + 1,
        ),
        "answer": answer,
    }


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={
        "ls_provider": PROVIDER_NAME_AGENT,
        "ls_model_name": MODEL_NAME_AGENT,
    },
)
def coordinator_agent(state: State) -> StateUpdate:

    template = prompt_template_config(
        "api/agents/prompts/coordinator_agent.yml", "coordinator_agent"
    )

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini", reasoning_effort="low", use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [FinalAgentResponse, Plan],
        tool_choice="required",
    )

    final_answer = False
    answer = ""
    response = llm_with_tools.invoke([SystemMessage(content=prompt), *state.messages])
    plan: List[Delegation] = []
    next_agent: AgentNode | None = None

    # Every request enters the graph at the coordinator, and `trace_id` identifies the
    # root run, so writing it here is enough to populate it for the whole turn. Later
    # coordinator passes rewrite the same value.
    trace_id = update_current_run_metadata(response)["trace_id"]

    if len(response.tool_calls) > 0:
        if response.tool_calls[0].get("name") == Plan.__name__:
            plan_response_validated = Plan.model_validate(
                response.tool_calls[0].get("args")
            )
            plan = plan_response_validated.plan
            next_agent = plan_response_validated.next_agent
            response = None
        else:
            for tool_call in response.tool_calls:
                if tool_call.get("name") == FinalAgentResponse.__name__:
                    final_answer = True
                    final_response_validated = FinalAgentResponse.model_validate(
                        tool_call.get("args")
                    )
                    answer = final_response_validated.answer
                    # FinalResponse is a structured-output "tool" that never receives a
                    # ToolMessage, so returning `response` as-is would persist an
                    # unanswered function call. On the next turn the checkpointer
                    # replays it and the Responses API rejects the request with
                    # "No tool output found for function call ...". Store a plain text
                    # AIMessage instead so the conversation history stays valid.
                    response = AIMessage(content=answer)
                    break

    return {
        "messages": [response] if response else [],
        "coordinator_agent": CoordinatorAgentProperties(
            final_answer=final_answer,
            iteration=state.coordinator_agent.iteration + 1,
            plan=plan,
            next_agent=next_agent,
        ),
        "trace_id": str(trace_id),
        "answer": answer,
    }


MetadataUpdateResponse = TypedDict(
    "MetadataUpdateResponse",
    {
        "trace_id": UUID,
    },
)


def update_current_run_metadata(response: AIMessage) -> MetadataUpdateResponse:
    current_run = get_current_run_tree()
    if not current_run:
        raise ValueError("No current run found")
    if not response.usage_metadata:
        raise ValueError("No usage metadata found in response")
    current_run.metadata["usage_metadata"] = {
        "input_tokens": response.usage_metadata["input_tokens"],
        "output_tokens": response.usage_metadata["output_tokens"],
        "total_tokens": response.usage_metadata["total_tokens"],
    }
    return {
        "trace_id": current_run.trace_id,
    }
