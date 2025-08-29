from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal

from models import GraphState
from nodes.check_normalize_address import check_normalize_address_llm
from nodes.find_address import find_address_llm
from nodes.web_search_address import web_search_address


def was_address_found(state: GraphState) -> Literal[END, "check_normalize_address_llm"]:
    """Check if any address was found"""
    if len(state['matchedAddresses']) > 0:
        return END
    else:
        return "check_normalize_address_llm"


# Add nodes and edges
builder = StateGraph(GraphState)
builder.add_node("web_search_address", web_search_address)
builder.add_node("find_address_llm", find_address_llm)
builder.add_node("check_normalize_address_llm", check_normalize_address_llm)

# Logic
builder.add_edge(START, "web_search_address")
builder.add_edge("web_search_address", "find_address_llm")
builder.add_conditional_edges("find_address_llm", was_address_found)
builder.add_edge("check_normalize_address_llm", END)

# Compile
graph = builder.compile()
