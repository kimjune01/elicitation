from langgraph.graph import StateGraph, END
from src.agent.state import PizzaState
from src.agent.nodes import extract_pizzas_node, gemini_llm, inspect_state_node, elicitation_response_node, order_confirmation_node, compute_pizza_completeness, human_node
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

GENERATE_PIZZAS = "extract_pizzas"
INSPECT_STATE = "inspect_state"
CHAT_INPUT = "chat_input"
ELICITATION_RESPONSE = "elicitation_response"
ORDER_CONFIRMATION = "order_confirmation"
HUMAN_NODE = "human_node"

def chat_input_node(inputs: Dict[str, Any]) -> PizzaState:
    # If already a PizzaState, just return it
    if isinstance(inputs, PizzaState):
        return inputs
    # Otherwise, treat as dict input
    messages = inputs.get('messages', [])
    print("CHAT INPUT messages:", messages)
    return PizzaState(messages=messages)

graph = StateGraph(state_schema=PizzaState)
graph.add_node(CHAT_INPUT, chat_input_node)
graph.add_node(GENERATE_PIZZAS, lambda state: extract_pizzas_node(state, gemini_llm))
graph.add_node(INSPECT_STATE, inspect_state_node)
graph.add_node(ELICITATION_RESPONSE, elicitation_response_node)
graph.add_node(ORDER_CONFIRMATION, order_confirmation_node)
graph.add_node(HUMAN_NODE, human_node)

graph.add_edge(CHAT_INPUT, GENERATE_PIZZAS)

def pizza_branching(state: PizzaState):
    _, incomplete_pizzas = compute_pizza_completeness(state)
    if incomplete_pizzas:
        return ELICITATION_RESPONSE
    else:
        return ORDER_CONFIRMATION

graph.add_conditional_edges(
    GENERATE_PIZZAS,
    pizza_branching,
    {
        ELICITATION_RESPONSE: ELICITATION_RESPONSE,
        ORDER_CONFIRMATION: ORDER_CONFIRMATION,
    }
)
graph.add_edge(ORDER_CONFIRMATION, INSPECT_STATE)

graph.add_edge(ELICITATION_RESPONSE, HUMAN_NODE)
graph.add_edge(HUMAN_NODE, CHAT_INPUT)

graph.add_edge(INSPECT_STATE, END)
graph.set_entry_point(CHAT_INPUT)

compiled_graph = graph.compile(name="Pizza Chat Graph")

# Export the compiled graph as 'graph' for langgraph.json
graph = compiled_graph
