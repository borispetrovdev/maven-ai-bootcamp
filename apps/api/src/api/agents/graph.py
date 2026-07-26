from enum import StrEnum
from typing import Any, Generator, Literal, assert_never, get_args

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import DebugPayload
from pydantic import BaseModel, TypeAdapter, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.agents.agents import agent_node, intent_router_node
from api.agents.retrieval_generation import (
    HYBRID_SEARCH_COLLECTION_NAME,
    RAGPipelineWithDecorationResponse,
    UsedContextEntry,
)
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context
from api.api.models import ItemPayload, State, StateUpdate


class Nodes(StrEnum):
    AGENT = "agent"
    INTENT_ROUTER = "intent_router"
    TOOLS = "tools"
    END = END


assert set(StateUpdate.__annotations__) == set(State.model_fields)

## Edges


def tool_router(state: State) -> Nodes:
    if state.final_answer:
        return Nodes.END
    if state.iteration > 2:
        return Nodes.END

    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return Nodes.TOOLS
    else:
        return Nodes.AGENT


def intent_router_conditional_edges(state: State) -> Nodes:
    if state.question_relevant:
        return Nodes.AGENT
    else:
        return Nodes.END


workflow = StateGraph(State)
tools = [get_formatted_item_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)

workflow.add_node(Nodes.TOOLS, tool_node)
workflow.add_node(Nodes.AGENT, agent_node)
workflow.add_node(Nodes.INTENT_ROUTER, intent_router_node)

workflow.add_edge(START, Nodes.INTENT_ROUTER)


workflow.add_conditional_edges(
    Nodes.INTENT_ROUTER,
    intent_router_conditional_edges,
    {
        Nodes.AGENT: Nodes.AGENT,
        Nodes.END: Nodes.END,
    },
)

workflow.add_conditional_edges(
    Nodes.AGENT, tool_router, {Nodes.TOOLS: Nodes.TOOLS, Nodes.END: END}
)

workflow.add_edge(Nodes.TOOLS, Nodes.AGENT)

workflow.add_edge(Nodes.AGENT, END)

graph = workflow.compile()


### Agent Execution


class AgentStreamData(RAGPipelineWithDecorationResponse):
    trace_id: str


class AgentStreamResponse(BaseModel):
    data: AgentStreamData
    type: Literal["final_answer"]


GraphStreamMode = Literal["values", "debug"]
"""The stream modes `agent_stream_wrapper` asks for and knows how to handle."""

STREAM_MODES: tuple[GraphStreamMode, ...] = get_args(GraphStreamMode)
"""Runtime counterpart of `GraphStreamMode`, derived so the two cannot drift apart."""

GraphStreamChunk = tuple[GraphStreamMode, Any]
"""A `(mode, payload)` pair, as yielded by `graph.stream` for a sequence `stream_mode`.

`Pregel.stream` annotates its return as `Iterator[dict[str, Any] | Any]`, which describes
neither the tuple nor the per-mode payloads, so the envelope is validated at runtime.
"""

_graph_stream_chunk_adapter = TypeAdapter(GraphStreamChunk)


def agent_stream_wrapper(question: str, thread_id: str) -> Generator[str, Any, Any]:

    def _string_for_sse(string: str) -> str:
        return f"data: {string}\n\n"

    def _status_for_debug_event(payload: DebugPayload[dict[str, Any]]) -> str | None:
        """Render a user-facing status line for a node that is about to run."""

        def _tool_to_text(tool_call: ToolCall) -> str | None:
            if tool_call["name"] == get_formatted_item_context.name:
                return f"Looking for items: {tool_call['args'].get('query', '')}."
            elif tool_call["name"] == get_formatted_reviews_context.name:
                return "Fetching user reviews..."
            return None

        if payload["type"] != "task":
            return None

        name = payload["payload"]["name"]
        if name == Nodes.INTENT_ROUTER:
            return "Analysing the question..."
        if name == Nodes.AGENT:
            return "Planning..."
        if name == Nodes.TOOLS:
            # TaskPayload.input is typed `Any` upstream, so validate before using it.
            task_input = State.model_validate(payload["payload"]["input"])
            last_message = task_input.messages[-1]
            if not isinstance(last_message, AIMessage):
                return None
            return " ".join(
                text
                for tool_call in last_message.tool_calls
                if (text := _tool_to_text(tool_call)) is not None
            )
        return None

    initial_state = State(messages=[HumanMessage(content=question)])
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result: State | None = None

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        for raw_chunk in graph.stream(
            initial_state,
            {"configurable": {"thread_id": thread_id}},
            # Must be a `list`: langgraph only yields `(mode, payload)` chunks behind an
            # `isinstance(stream_mode, list)` check, so a tuple yields bare payloads instead.
            stream_mode=list(STREAM_MODES),
        ):
            mode, payload = _graph_stream_chunk_adapter.validate_python(raw_chunk)

            if mode == "debug":
                status = _status_for_debug_event(payload)
                if status:
                    yield _string_for_sse(status)
            elif mode == "values":
                result = State.model_validate(payload)
            else:
                assert_never(mode)

    used_context: list[UsedContextEntry] = []
    for item in result.references if result else []:
        points = qdrant_client.scroll(
            collection_name=HYBRID_SEARCH_COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
                ]
            ),
        )[0]

        if len(points) == 0:
            continue

        payload = points[0].payload
        if not payload:
            raise ValueError(f"No payload in point: {points[0].id}")
        try:
            payload = ItemPayload.model_validate(payload)
        except ValidationError as e:
            raise ValueError(f"Invalid payload: {payload}, error: {e}") from e
        used_context.append(
            {
                "id": item.id,
                "image_url": str(payload.image),
                "price": payload.price,
                "description": item.description,
            }
        )

    if result:
        to_serialize: AgentStreamData = AgentStreamData(
            answer=result.answer,
            used_context=used_context,
            trace_id=result.trace_id,
        )
        yield _string_for_sse(
            (
                AgentStreamResponse(data=to_serialize, type="final_answer")
            ).model_dump_json()
        )
