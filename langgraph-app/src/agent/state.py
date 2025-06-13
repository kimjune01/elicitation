from typing import Optional, List, Literal, Tuple, Dict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

PizzaCrust = Literal['thin', 'classic', 'stuffed']
PizzaSize = Literal['small', 'medium', 'large', 'extra_large']
PizzaTopping = Literal[
    'pepperoni', 'mushrooms', 'onions', 'sausage', 'bacon', 'cheese',
    'extra cheese', 'black olives', 'green peppers', 'pineapple', 'spinach',
    'ham', 'tomatoes', 'chicken', 'beef', 'anchovies',
    'jalapenos', 'garlic', 'artichokes', 'broccoli', 'feta cheese',
    'salami', 'red onions', 'corn', 'zucchini', 'eggplant',
    'prosciutto', 'basil', 'sun-dried tomatoes', 'roasted red peppers', 'arugula'
]

class Pizza(BaseModel):
    crust: Optional[PizzaCrust] = None
    toppings: Optional[List[PizzaTopping]] = None
    size: Optional[PizzaSize] = None

class PizzaState(BaseModel):
    pizzas: List[Pizza] = Field(default_factory=list)
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)
    ambiguous: List[Tuple[int, str]] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

def create_initial_state(pizzas: List[Pizza], rejected: Optional[List[str]] = None, ambiguous: Optional[List[Tuple[int, str]]] = None, errors: Optional[List[str]] = None) -> PizzaState:
    pizzas_with_cheese = []
    for pizza in pizzas:
        # Only toppings is defaulted; crust and size remain None if not specified
        if isinstance(pizza, dict):
            pizza_obj = Pizza(**pizza)
        else:
            pizza_obj = pizza
            
        toppings = pizza_obj.toppings
        if toppings is None:
            pizza_obj.toppings = ['cheese']
        elif 'cheese' not in toppings:
            pizza_obj.toppings = toppings + ['cheese']
        # Do NOT default crust or size; leave as None if not present
        pizzas_with_cheese.append(pizza_obj)
    
    return PizzaState(
        pizzas=pizzas_with_cheese,
        messages=[],
        rejected=rejected or [],
        ambiguous=ambiguous or [],
        questions=[],
        errors=errors or []
    )
