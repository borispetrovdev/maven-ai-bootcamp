from enum import StrEnum
from typing import Any, Generator, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ValidationError
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


def agent_stream_wrapper(question: str, thread_id: str) -> Generator[str, Any, Any]:

    def _string_for_sse(string: str) -> str:
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):
        # print(chunk)

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == get_formatted_item_context.name:
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == get_formatted_reviews_context.name:
                return "Fetching user reviews..."

        if _is_node_start(chunk):
            name = chunk[1].get("payload", {}).get("name")
            if name == Nodes.INTENT_ROUTER:
                return "Analysing the question..."
            if name == Nodes.AGENT:
                return "Planning..."
            if name == Nodes.TOOLS:
                message = " ".join(
                    text
                    for tool_call in chunk[1]
                    .get("payload", {})
                    .get("input", {})
                    .messages[-1]
                    .tool_calls
                    if (text := _tool_to_text(tool_call)) is not None
                )
                return message

    initial_state = State(messages=[HumanMessage(content=question)])
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result: State | None = None

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state,
            {"configurable": {"thread_id": thread_id}},
            stream_mode=["values", "debug"],
        ):
            processed_chunk = _process_graph_event(chunk)
            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = State.model_validate(chunk[1])

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
