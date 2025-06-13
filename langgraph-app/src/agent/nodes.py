import os
import google.generativeai as genai
from pydantic import BaseModel
from src.agent.prompts import PIZZA_EXTRACTION_PROMPT, ORDER_SUMMARY_PROMPT
from src.agent.state import Pizza, PizzaState, create_initial_state
from typing import List, Tuple, TypedDict, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json
from langgraph.types import interrupt

# Define Pydantic models for structured output
class PizzaModel(BaseModel):
    crust: str
    toppings: List[str]
    size: str

class PizzaExtractionResult(BaseModel):
    pizzas: List[PizzaModel]
    rejected: List[str]
    ambiguous: List[List]

# DIY LLM function using Gemini with structured output
def gemini_llm(prompt_text, config={}):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=prompt_text,
            config=config,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[gemini_llm] Structured output failed: {e}. Trying plain text fallback.")
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        response = model.generate_content(prompt_text)
        return response.text.strip()

def validate_pizzas(state: PizzaState) -> PizzaState:
    questions = []
    for idx, pizza in enumerate(state.pizzas):
        if not pizza.crust:
            questions.append(f"What crust for pizza #{idx+1}?")
        if not pizza.toppings:
            questions.append(f"What toppings for pizza #{idx+1}?")
        if not pizza.size:
            questions.append(f"What size for pizza #{idx+1}?")
    state.questions = questions
    return state

def generate_ambiguities(state: PizzaState) -> List[Tuple[int, str]]:
    ambiguities = []
    for idx, pizza in enumerate(state.pizzas):
        if pizza.crust is None:
            ambiguities.append((idx, 'crust'))
        if pizza.toppings is None:
            ambiguities.append((idx, 'toppings'))
        if pizza.size is None:
            ambiguities.append((idx, 'size'))
    return ambiguities

def format_messages(messages):
    """
    Formats a messages list into a string with each message on a new line.
    Example:
        Input:
            [HumanMessage(content="I want a pizza with mushrooms."),
             AIMessage(content="What size would you like?")]
        Output:
            "human: I want a pizza with mushrooms.\nai: What size would you like?"
    """
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"ai: {msg.content}")
        elif isinstance(msg, dict):
            # Handle legacy dict format for backwards compatibility
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")
        else:
            # Generic BaseMessage handling
            role = msg.__class__.__name__.lower().replace('message', '')
            formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def build_pizza_extraction_prompt(messages: list) -> str:
    """
    Build the pizza extraction prompt from a messages list.
    """
    messages_str = format_messages(messages)
    return PIZZA_EXTRACTION_PROMPT.format(messages=messages_str)

def parse_llm_pizza_response(response: str) -> tuple:
    """
    Parse the LLM response for pizza extraction.
    Returns (pizzas, rejected, ambiguous, errors)
    Handles code block formatted JSON (```json ... ```)
    """
    errors = []
    pizzas, rejected, ambiguous = [], [], []
    response = response.strip()
    # Remove code block formatting if present
    if response.startswith('```json'):
        response = response[len('```json'):].strip()
    if response.startswith('```'):
        response = response[len('```'):].strip()
    if response.endswith('```'):
        response = response[:-3].strip()
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            normalized = {k.strip(): v for k, v in result.items()}
            pizzas = normalized.get('pizzas', [])
            rejected = normalized.get('rejected', [])
            ambiguous = normalized.get('ambiguous', [])
            # Convert empty string fields to None for Pydantic compatibility
            for pizza in pizzas:
                if 'crust' in pizza and pizza['crust'] == '':
                    pizza['crust'] = None
                if 'size' in pizza and pizza['size'] == '':
                    pizza['size'] = None
                if 'toppings' in pizza and pizza['toppings'] == '':
                    pizza['toppings'] = None
        else:
            errors.append(f"LLM response JSON is not an object: {type(result).__name__}")
            errors.append(f"Raw response: {response}")
    except Exception as e:
        errors.append(f"Failed to parse LLM response as JSON: {str(e)}")
        errors.append(f"Raw response: {response}")
    return pizzas, rejected, ambiguous, errors

def extract_pizzas_node(state: PizzaState) -> PizzaState:
    """
    Node to extract pizzas from the messages using the provided LLM.
    """
    prompt = build_pizza_extraction_prompt(state.messages)
    errors = []
    try:
        response = gemini_llm(prompt, config={
            "response_mime_type": "application/json",
            "response_schema": PizzaExtractionResult,
        })
        pizzas, rejected, ambiguous, parse_errors = parse_llm_pizza_response(response)
        errors.extend(parse_errors)
    except Exception as e:
        pizzas, rejected, ambiguous = [], [], []
        errors.append(f"LLM call failed: {str(e)}")
    new_state = create_initial_state(pizzas, rejected=rejected, ambiguous=ambiguous, errors=errors)
    new_state.messages = state.messages
    return new_state

def inspect_state_node(state):
    print("INSPECT STATE:", state)
    # If there are errors, print any raw LLM output for debugging
    if state.errors:
        for error in state.errors:
            if error.startswith("Raw response:"):
                print("RAW LLM OUTPUT:", error[len("Raw response:"):].strip())
    return state

def compute_pizza_completeness(state: PizzaState):
    """
    Returns (complete_pizzas, incomplete_pizzas) from the pizzas array and ambiguous list.
    complete_pizzas: list of (idx, pizza) tuples
    incomplete_pizzas: list of dicts with index, pizza, missing_fields, ambiguous_fields
    """
    complete_pizzas = []
    incomplete_pizzas = []
    for idx, pizza in enumerate(state.pizzas):
        missing_fields = []
        if not pizza.crust:
            missing_fields.append('crust')
        if not pizza.toppings:
            missing_fields.append('toppings')
        if not pizza.size:
            missing_fields.append('size')
        ambiguous_fields = [amb[1] for amb in state.ambiguous if amb[0] == idx]
        if not missing_fields and not ambiguous_fields:
            complete_pizzas.append((idx, pizza))
        else:
            incomplete_pizzas.append({
                'index': ordinal(idx+1),
                'pizza': pizza,
                'accepted_fields': {field: getattr(pizza, field) for field in ['crust', 'toppings', 'size'] if getattr(pizza, field)},
                'missing_fields': missing_fields,
                'ambiguous_fields': ambiguous_fields
            })
    return complete_pizzas, incomplete_pizzas

def ordinal(n):
    # Returns 'first', 'second', ... for 1-based n
    ordinals = [
        'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
        'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth'
    ]
    if 1 <= n <= len(ordinals):
        return ordinals[n-1]
    return f"{n}th"

def make_accepted_fields(incomplete_pizzas: List[Dict]) -> List[Dict]:
    """
    For each incomplete pizza, return a dict of accepted fields and their values,
    constructed from the pizza object.
    """
    accepted = []
    for inc in incomplete_pizzas:
        pizza = inc.get('pizza')
        if pizza:
            accepted_fields = {field: getattr(pizza, field) for field in ['crust', 'toppings', 'size'] if getattr(pizza, field)}
            accepted.append(accepted_fields)
    return accepted

def elicitation_response_node(state: PizzaState) -> PizzaState:
    """
    Node to generate a response asking for missing or ambiguous pizza properties using ORDER_SUMMARY_PROMPT.
    """
    complete_pizzas, incomplete_pizzas = compute_pizza_completeness(state)
    accepted = make_accepted_fields(incomplete_pizzas)
    rejected = state.rejected
    ambiguous = state.ambiguous
    missing: List[str] = []
    for inc in incomplete_pizzas:
        missing.extend(inc['missing_fields'])
    # Format for prompt
    accepted_str = str(accepted)
    rejected_str = str(rejected)
    ambiguous_str = str(ambiguous)
    missing_str = str(missing)
    complete_pizzas_str = str(complete_pizzas)
    prompt = ORDER_SUMMARY_PROMPT.format(
        accepted=accepted_str,
        rejected=rejected_str,
        ambiguous=ambiguous_str,
        missing=missing_str,
        complete_pizzas=complete_pizzas_str
    )
    response = gemini_llm(prompt)
    state.messages.append(AIMessage(content=response))
    return state

def order_confirmation_node(state: PizzaState) -> PizzaState:
    """
    Node to confirm the order is complete and ready for processing.
    """
    complete_pizzas, incomplete_pizzas = compute_pizza_completeness(state)
    assert not incomplete_pizzas, "order_confirmation_node called with incomplete pizzas!"
    # Format the complete pizzas for the confirmation message
    pizza_descriptions = []
    for idx, pizza in complete_pizzas:
        pizza_num = ordinal(idx+1)
        desc = f"Your {pizza_num} pizza: {pizza.size or '?'} {pizza.crust or '?'} crust with {', '.join(pizza.toppings or [])}"
        pizza_descriptions.append(desc)
    pizzas_str = '\n'.join(pizza_descriptions)
    confirmation_message = f"Your pizza order is complete!\n\nOrder summary:\n{pizzas_str}\n\nThank you."
    state.messages.append(AIMessage(content=confirmation_message))
    return state

def human_node(state: PizzaState) -> PizzaState:
    """
    Node that represents a user input loop. No interrupt, just returns state.
    """
    return state

