from operator import add
from typing import Annotated, List, Literal, Optional, TypedDict, Union

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, HttpUrl

from api.agents.nodes import AgentNode


class ItemPayload(BaseModel):
    preprocessed_description: str = Field(description="Cleaned item description text")
    image: HttpUrl = Field(description="Image URL of the item")
    rating_number: int = Field(ge=0, description="Number of ratings")
    price: float | None = Field(default=None, ge=0, description="Item price")
    average_rating: float = Field(ge=0, le=5, description="Average rating (0-5)")
    parent_asin: str = Field(description="Amazon parent ASIN identifier")


class ProductInDb(TypedDict):
    product_id: str
    quantity: int


class ShoppingCartItem(ProductInDb):
    # `shopping_cart_items.price` is nullable and `ItemPayload.price` is optional, so
    # items whose Qdrant payload carries no price land in the cart with a NULL price —
    # and `total_price` is `price * quantity`, so it is NULL right alongside it.
    price: float | None
    currency: str
    product_image_url: str
    total_price: float | None


class AgentRequest(BaseModel):
    query: str
    thread_id: str


class RAGUsedContextSimple(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(
        description="Description of the item corresponding to the id"
    )


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    image_url: str = Field(description="Image URL of the item corresponding to the id")
    price: Optional[float] = Field(
        default=None, description="Price of the item corresponding to the id"
    )
    description: str = Field(
        description="Description of the item corresponding to the id"
    )


class AgentResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext] = Field(
        description="List of items used to answer the question"
    )
    trace_id: str = Field(description="Trace ID of the run")
    shopping_cart: list[ShoppingCartItem] = Field(
        description="List of products in the shopping cart"
    )


class Delegation(BaseModel):
    agent: AgentNode = Field(description="The agent to delegate the task to")
    task: str = Field(description="The task to be performed by the agent")


class Plan(BaseModel):
    next_agent: AgentNode = Field(description="The next agent to invoke")
    plan: List[Delegation] = Field(
        description="A list of delegations to agents with tasks to be performed in sequence."
    )


class FinalQnAAgentResponse(BaseModel):
    """Call this tool when the final answer is possible using available context."""

    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContextSimple] = Field(
        description="List of items used to answer the question"
    )


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False


class CoordinatorAgentProperties(AgentProperties):
    plan: List[Delegation] = []
    next_agent: AgentNode | None = None


type UserIntent = Literal["product_qna", "shopping_cart", "other"]


class StateUpdate(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add]
    product_qna_agent: AgentProperties
    shopping_cart_agent: AgentProperties
    warehouse_manager_agent: AgentProperties
    user_intent: UserIntent
    answer: str
    user_id: str
    cart_id: str
    references: list[RAGUsedContextSimple]
    coordinator_agent: CoordinatorAgentProperties
    trace_id: str


class State(BaseModel):
    messages: Annotated[List[AnyMessage], add] = []
    user_intent: UserIntent | None = None
    product_qna_agent: AgentProperties = AgentProperties()
    shopping_cart_agent: AgentProperties = AgentProperties()
    warehouse_manager_agent: AgentProperties = AgentProperties()
    answer: str = ""
    user_id: str
    cart_id: str
    references: list[RAGUsedContextSimple] = []
    coordinator_agent: CoordinatorAgentProperties = CoordinatorAgentProperties()
    trace_id: str = ""


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_score: Union[int, None] = Field(description="Feedback score, 0 or 1")
    feedback_text: str = Field(description="Feedback text")
    feedback_source_type: Literal["api", "user"] = Field(
        description="Feedback source type, api or user"
    )


class FeedbackResponse(BaseModel):
    message: str = Field(description="Message of the feedback submission")


class FinalAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question")


assert set(StateUpdate.__annotations__).issubset(State.model_fields)
