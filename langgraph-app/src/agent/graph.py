from langgraph.graph import StateGraph, END
from agent.state import PizzaState, Message
from agent.nodes import extract_pizzas_node, gemini_llm, inspect_state_node
from typing import List, Dict, Any

GENERATE_PIZZAS = "extract_pizzas"
INSPECT_STATE = "inspect_state"
CHAT_INPUT = "chat_input"

def chat_input_node(inputs: Dict[str, Any]) -> PizzaState:
    # If already a PizzaState, just return it
    if isinstance(inputs, PizzaState):
        return inputs
    # Otherwise, treat as dict input
    messages = inputs.get('messages', [])
    conversation: List[Message] = [
        {"role": msg["role"], "content": msg["content"]} for msg in messages
    ]
    return PizzaState(conversation=conversation)

graph = StateGraph(state_schema=PizzaState)
graph.add_node(CHAT_INPUT, chat_input_node)
graph.add_node(GENERATE_PIZZAS, lambda state: extract_pizzas_node(state, gemini_llm))
graph.add_node(INSPECT_STATE, inspect_state_node)

graph.add_edge(CHAT_INPUT, GENERATE_PIZZAS)
graph.add_edge(GENERATE_PIZZAS, INSPECT_STATE)
graph.add_edge(INSPECT_STATE, END)
graph.set_entry_point(CHAT_INPUT)

compiled_graph = graph.compile(name="Pizza Chat Graph")

result = compiled_graph.invoke({"messages": [{"role": "caller", "content": "I want a pizza with mushrooms."}]})
print(result)
