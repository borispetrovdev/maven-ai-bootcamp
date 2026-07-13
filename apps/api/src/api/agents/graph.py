from enum import StrEnum

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from api.agents.agents import agent_node, intent_router_node
from api.agents.retrieval_generation import (
    HYBRID_SEARCH_COLLECTION_NAME,
    RAGPipelineWithDecorationResponse,
    UsedContextEntry,
)
from api.agents.tools import get_formatted_item_context
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
tools = [get_formatted_item_context]
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


def run_agent(question: str) -> State:
    initial_state = State(messages=[HumanMessage(content=question)])
    result = graph.invoke(initial_state)
    return State.model_validate(result)


def agent_wrapper(question: str) -> RAGPipelineWithDecorationResponse:
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = run_agent(question)

    used_context: list[UsedContextEntry] = []
    for item in result.references:
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

    return {
        "answer": result.answer,
        "used_context": used_context,
    }
