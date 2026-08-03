from typing import Any, Generator, Literal, assert_never, get_args

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import DebugPayload
from pydantic import BaseModel, TypeAdapter, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.agents.agents import (
    coordinator_agent,
    product_qna_agent,
    shopping_cart_agent,
    warehouse_manager_agent,
)
from api.agents.nodes import Nodes
from api.agents.retrieval_generation import (
    HYBRID_SEARCH_COLLECTION_NAME,
    RAGPipelineWithDecorationResponse,
    UsedContextEntry,
)
from api.agents.tools import (
    add_to_shopping_cart,
    check_warehouse_availability,
    get_formatted_item_context,
    get_formatted_reviews_context,
    get_shopping_cart,
    get_shopping_cart_nontool,
    remove_from_cart,
    reserve_warehouse_items,
)
from api.api.models import (
    AgentProperties,
    CoordinatorAgentProperties,
    ItemPayload,
    ShoppingCartItem,
    State,
    StateUpdate,
)

## Edges


def product_qna_agent_tool_router(
    state: State,
) -> Literal[Nodes.END, Nodes.PRODUCT_QNA_TOOLS]:
    if state.product_qna_agent.final_answer:
        return Nodes.END
    if state.product_qna_agent.iteration > 5:
        return Nodes.END
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return Nodes.PRODUCT_QNA_TOOLS
    return Nodes.END


def shopping_cart_agent_tool_router(
    state: State,
) -> Literal[Nodes.END, Nodes.SHOPPING_CART_TOOLS]:
    if state.shopping_cart_agent.final_answer:
        return Nodes.END
    if state.shopping_cart_agent.iteration > 4:
        return Nodes.END
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return Nodes.SHOPPING_CART_TOOLS
    return Nodes.END


def warehouse_manager_agent_tool_router(
    state: State,
) -> Literal[Nodes.END, Nodes.WAREHOUSE_MANAGER_TOOLS]:
    if state.warehouse_manager_agent.final_answer:
        return Nodes.END
    if state.warehouse_manager_agent.iteration > 4:
        return Nodes.END
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return Nodes.WAREHOUSE_MANAGER_TOOLS
    return Nodes.END


def coordinator_agent_edge(state: State) -> Nodes:
    if state.coordinator_agent.final_answer:
        return Nodes.END
    if state.coordinator_agent.iteration > 6:
        return Nodes.END
    if state.coordinator_agent.next_agent == Nodes.PRODUCT_QNA_AGENT:
        return Nodes.PRODUCT_QNA_AGENT
    elif state.coordinator_agent.next_agent == Nodes.SHOPPING_CART_AGENT:
        return Nodes.SHOPPING_CART_AGENT
    elif state.coordinator_agent.next_agent == Nodes.WAREHOUSE_MANAGER_AGENT:
        return Nodes.WAREHOUSE_MANAGER_AGENT
    elif state.coordinator_agent.next_agent is None:
        raise RuntimeError("Coordinator neither delegated nor answered")
    else:
        assert_never(state.coordinator_agent.next_agent)


# `input_schema` declares that graph inputs are a *partial* update, not a whole `State`.
# Passing a `State` instance would write every field, including ones left at their
# defaults, wiping values the checkpointer holds from earlier turns (`references` is the
# one that bites: only `product_qna_agent` sets it, so any turn routed elsewhere would
# blank the UI's suggestions). A `total=False` TypedDict lets keys be genuinely absent.
workflow = StateGraph(State, input_schema=StateUpdate)
product_qna_agent_tools = [get_formatted_item_context, get_formatted_reviews_context]
shopping_cart_agent_tools = [get_shopping_cart, add_to_shopping_cart, remove_from_cart]
warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
product_qna_agent_tool_node = ToolNode(product_qna_agent_tools)
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)
warehouse_manager_agent_tool_node = ToolNode(warehouse_manager_agent_tools)

workflow.add_node(Nodes.PRODUCT_QNA_TOOLS, product_qna_agent_tool_node)
workflow.add_node(Nodes.SHOPPING_CART_TOOLS, shopping_cart_agent_tool_node)
workflow.add_node(Nodes.WAREHOUSE_MANAGER_TOOLS, warehouse_manager_agent_tool_node)
workflow.add_node(Nodes.PRODUCT_QNA_AGENT, product_qna_agent)
workflow.add_node(Nodes.SHOPPING_CART_AGENT, shopping_cart_agent)
workflow.add_node(Nodes.WAREHOUSE_MANAGER_AGENT, warehouse_manager_agent)
workflow.add_node(Nodes.COORDINATOR_AGENT, coordinator_agent)

workflow.add_edge(START, Nodes.COORDINATOR_AGENT)

workflow.add_conditional_edges(
    Nodes.COORDINATOR_AGENT,
    coordinator_agent_edge,
    {
        Nodes.PRODUCT_QNA_AGENT: Nodes.PRODUCT_QNA_AGENT,
        Nodes.SHOPPING_CART_AGENT: Nodes.SHOPPING_CART_AGENT,
        Nodes.WAREHOUSE_MANAGER_AGENT: Nodes.WAREHOUSE_MANAGER_AGENT,
        Nodes.END: Nodes.END,
    },
)

workflow.add_conditional_edges(
    Nodes.PRODUCT_QNA_AGENT,
    product_qna_agent_tool_router,
    {
        Nodes.PRODUCT_QNA_TOOLS: Nodes.PRODUCT_QNA_TOOLS,
        Nodes.END: Nodes.COORDINATOR_AGENT,
    },
)
workflow.add_conditional_edges(
    Nodes.SHOPPING_CART_AGENT,
    shopping_cart_agent_tool_router,
    {
        Nodes.SHOPPING_CART_TOOLS: Nodes.SHOPPING_CART_TOOLS,
        Nodes.END: Nodes.COORDINATOR_AGENT,
    },
)

workflow.add_conditional_edges(
    Nodes.WAREHOUSE_MANAGER_AGENT,
    warehouse_manager_agent_tool_router,
    {
        Nodes.WAREHOUSE_MANAGER_TOOLS: Nodes.WAREHOUSE_MANAGER_TOOLS,
        Nodes.END: Nodes.COORDINATOR_AGENT,
    },
)

workflow.add_edge(Nodes.PRODUCT_QNA_TOOLS, Nodes.PRODUCT_QNA_AGENT)
workflow.add_edge(Nodes.SHOPPING_CART_TOOLS, Nodes.SHOPPING_CART_AGENT)
workflow.add_edge(Nodes.WAREHOUSE_MANAGER_TOOLS, Nodes.WAREHOUSE_MANAGER_AGENT)

### Agent Execution


class AgentStreamData(RAGPipelineWithDecorationResponse):
    trace_id: str
    shopping_cart: list[ShoppingCartItem]


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

_TOOL_STATUS_TEXT: dict[str, str] = {
    get_formatted_reviews_context.name: "Fetching user reviews...",
    get_shopping_cart.name: "Fetching shopping cart...",
    add_to_shopping_cart.name: "Adding items to shopping cart...",
    remove_from_cart.name: "Removing items from shopping cart...",
    check_warehouse_availability.name: "Checking warehouse availability...",
    reserve_warehouse_items.name: "Reserving warehouse items...",
}
"""Status line per tool, keyed off each tool's `.name` so the keys track the tool
definitions instead of drifting from hand-written literals. `get_formatted_item_context`
interpolates its query, so it is handled separately."""


def agent_stream_wrapper(question: str, thread_id: str) -> Generator[str, Any, Any]:

    def _string_for_sse(string: str) -> str:
        return f"data: {string}\n\n"

    def _status_for_debug_event(payload: DebugPayload[dict[str, Any]]) -> str | None:
        """Render a user-facing status line for a node that is about to run."""

        def _tool_to_text(tool_call: ToolCall) -> str | None:
            if tool_call["name"] == get_formatted_item_context.name:
                return f"Looking for items: {tool_call['args'].get('query', '')}."
            return _TOOL_STATUS_TEXT.get(tool_call["name"])

        if payload["type"] != "task":
            return None

        try:
            node = Nodes(payload["payload"]["name"])
        except ValueError:
            # langgraph's own nodes (`__start__`, …) are not in `Nodes` and get no status.
            return None

        match node:
            case Nodes.COORDINATOR_AGENT:
                return "Analysing the question..."
            case (
                Nodes.PRODUCT_QNA_AGENT
                | Nodes.SHOPPING_CART_AGENT
                | Nodes.WAREHOUSE_MANAGER_AGENT
            ):
                return "Planning..."
            case (
                Nodes.PRODUCT_QNA_TOOLS
                | Nodes.SHOPPING_CART_TOOLS
                | Nodes.WAREHOUSE_MANAGER_TOOLS
            ):
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
            case Nodes.END:
                return None
            case _:
                assert_never(node)

    initial_state: StateUpdate = {
        "messages": [HumanMessage(content=question)],
        "user_id": thread_id,
        "cart_id": thread_id,
        "coordinator_agent": CoordinatorAgentProperties(),
        "product_qna_agent": AgentProperties(),
        "shopping_cart_agent": AgentProperties(),
        "warehouse_manager_agent": AgentProperties(),
    }
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
        shopping_cart = get_shopping_cart_nontool(result.user_id, result.cart_id)

        to_serialize: AgentStreamData = AgentStreamData(
            answer=result.answer,
            used_context=used_context,
            trace_id=result.trace_id,
            shopping_cart=shopping_cart,
        )
        yield _string_for_sse(
            (
                AgentStreamResponse(data=to_serialize, type="final_answer")
            ).model_dump_json()
        )
