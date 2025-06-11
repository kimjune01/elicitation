import os
import google.generativeai as genai
from pydantic import BaseModel
from agent.prompts import PIZZA_EXTRACTION_PROMPT
from agent.state import Pizza, PizzaState, create_initial_state
from typing import List, Tuple, TypedDict
import json

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
            model="gemini-2.0-flash",
            contents=prompt_text,
            config=config,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[gemini_llm] Structured output failed: {e}. Trying plain text fallback.")
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt_text)
        return response.text.strip()

def validate_pizzas(state: PizzaState) -> PizzaState:
    questions = []
    for idx, pizza in enumerate(state.pizzas):
        if not pizza.get('crust'):
            questions.append(f"What crust for pizza #{idx+1}?")
        if not pizza.get('toppings'):
            questions.append(f"What toppings for pizza #{idx+1}?")
        if not pizza.get('size'):
            questions.append(f"What size for pizza #{idx+1}?")
    state.questions = questions
    return state

def generate_ambiguities(state: PizzaState) -> List[Tuple[int, str]]:
    ambiguities = []
    for idx, pizza in enumerate(state.pizzas):
        for field in ['crust', 'toppings', 'size']:
            if pizza.get(field) is None:
                ambiguities.append((idx, field))
    return ambiguities

def format_conversation(conversation):
    """
    Formats a conversation (list of Message dicts) into a string with each message on a new line.
    Example:
        Input:
            [
                {"role": "caller", "content": "I want a pizza with mushrooms."},
                {"role": "receiver", "content": "What size would you like?"}
            ]
        Output:
            "caller: I want a pizza with mushrooms.\nreceiver: What size would you like?"
    """
    return "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in conversation
    ])

def build_pizza_extraction_prompt(conversation: list) -> str:
    """
    Build the pizza extraction prompt from a conversation list.
    """
    conversation_str = format_conversation(conversation)
    return PIZZA_EXTRACTION_PROMPT.format(conversation=conversation_str)

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
        else:
            errors.append(f"LLM response JSON is not an object: {type(result).__name__}")
            errors.append(f"Raw response: {response}")
    except Exception as e:
        errors.append(f"Failed to parse LLM response as JSON: {str(e)}")
        errors.append(f"Raw response: {response}")
    return pizzas, rejected, ambiguous, errors

def extract_pizzas_node(state: PizzaState, llm) -> PizzaState:
    """
    Node to extract pizzas from the conversation using the provided LLM.
    """
    prompt = build_pizza_extraction_prompt(state.conversation)
    errors = []
    try:
        response = gemini_llm(prompt, config={
            "response_mime_type": "application/json",
            "response_schema": PizzaExtractionResult,
        })
        print("response:", response)
        pizzas, rejected, ambiguous, parse_errors = parse_llm_pizza_response(response)
        errors.extend(parse_errors)
    except Exception as e:
        pizzas, rejected, ambiguous = [], [], []
        errors.append(f"LLM call failed: {str(e)}")
    new_state = create_initial_state(pizzas, rejected=rejected, ambiguous=ambiguous, errors=errors)
    new_state.conversation = state.conversation
    print("NEW STATE:", new_state)
    return new_state

def inspect_state_node(state):
    print("INSPECT STATE:", state)
    # If there are errors, print any raw LLM output for debugging
    if hasattr(state, 'errors') and state.errors:
        for error in state.errors:
            if error.startswith("Raw response:"):
                print("RAW LLM OUTPUT:", error[len("Raw response:"):].strip())
    return state

