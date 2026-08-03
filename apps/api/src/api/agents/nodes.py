"""Graph node names, kept in a leaf module.

`Nodes` and `AgentNode` name the graph's topology, but the Pydantic models in
`api.api.models` need `AgentNode` to type the coordinator's delegation targets, and
`api.agents.agents` needs it too. Defining them alongside the graph itself would make
`graph` -> `models` -> `graph` a cycle, so they live here and depend on nothing internal.
"""

from enum import StrEnum
from typing import Literal

from langgraph.graph import END


class Nodes(StrEnum):
    PRODUCT_QNA_AGENT = "product_qna_agent"
    SHOPPING_CART_AGENT = "shopping_cart_agent"
    WAREHOUSE_MANAGER_AGENT = "warehouse_manager_agent"
    COORDINATOR_AGENT = "coordinator_agent"
    PRODUCT_QNA_TOOLS = "product_qna_tools"
    SHOPPING_CART_TOOLS = "shopping_cart_tools"
    WAREHOUSE_MANAGER_TOOLS = "warehouse_manager_tools"
    END = END


type AgentNode = Literal[
    Nodes.PRODUCT_QNA_AGENT, Nodes.SHOPPING_CART_AGENT, Nodes.WAREHOUSE_MANAGER_AGENT
]
