"""
Graph wrapper specifically for LangGraph Studio compatibility.
"""
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.agent.state import PizzaState
from src.agent.nodes import extract_pizzas_node, gemini_llm, inspect_state_node, elicitation_response_node, order_confirmation_node, compute_pizza_completeness

class StudioState(TypedDict):
    """State that works with LangGraph Studio's expected interface"""
    messages: List[Dict[str, str]]

def convert_to_pizza_state(studio_state: Dict[str, Any]) -> PizzaState:
    """Convert from Studio state to internal PizzaState"""
    messages = studio_state.get('messages', [])
    messages_list: List[Dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]} for msg in messages
    ]
    return PizzaState(messages=messages_list)

def convert_from_pizza_state(pizza_state: PizzaState) -> Dict[str, Any]:
    """Convert from internal PizzaState back to Studio state"""
    return {
        "messages": pizza_state.messages
    }

# Node wrappers that handle conversion
def studio_chat_input(state: Dict[str, Any]) -> Dict[str, Any]:
    pizza_state = convert_to_pizza_state(state)
    return convert_from_pizza_state(pizza_state)

def studio_extract_pizzas(state: Dict[str, Any]) -> Dict[str, Any]:
    pizza_state = convert_to_pizza_state(state)
    updated_state = extract_pizzas_node(pizza_state, gemini_llm)
    return convert_from_pizza_state(updated_state)

def studio_elicitation_response(state: Dict[str, Any]) -> Dict[str, Any]:
    pizza_state = convert_to_pizza_state(state)
    updated_state = elicitation_response_node(pizza_state)
    return convert_from_pizza_state(updated_state)

def studio_order_confirmation(state: Dict[str, Any]) -> Dict[str, Any]:
    pizza_state = convert_to_pizza_state(state)
    updated_state = order_confirmation_node(pizza_state)
    return convert_from_pizza_state(updated_state)

def studio_inspect_state(state: Dict[str, Any]) -> Dict[str, Any]:
    pizza_state = convert_to_pizza_state(state)
    inspect_state_node(pizza_state)
    return state

def studio_pizza_branching(state: Dict[str, Any]) -> str:
    pizza_state = convert_to_pizza_state(state)
    _, incomplete_pizzas = compute_pizza_completeness(pizza_state)
    if incomplete_pizzas:
        return "elicitation_response"
    else:
        return "order_confirmation"

# Create the graph with Studio-compatible state
studio_graph = StateGraph(state_schema=StudioState)

# Add nodes
studio_graph.add_node("chat_input", studio_chat_input)
studio_graph.add_node("extract_pizzas", studio_extract_pizzas)
studio_graph.add_node("elicitation_response", studio_elicitation_response)
studio_graph.add_node("order_confirmation", studio_order_confirmation)
studio_graph.add_node("inspect_state", studio_inspect_state)

# Add edges
studio_graph.add_edge("chat_input", "extract_pizzas")
studio_graph.add_conditional_edges(
    "extract_pizzas",
    studio_pizza_branching,
    {
        "elicitation_response": "elicitation_response",
        "order_confirmation": "order_confirmation",
    }
)
studio_graph.add_edge("elicitation_response", "inspect_state")
studio_graph.add_edge("order_confirmation", "inspect_state")
studio_graph.add_edge("inspect_state", END)

# Set entry point
studio_graph.set_entry_point("chat_input")

# Compile and export
graph = studio_graph.compile(name="Pizza Chat Studio")