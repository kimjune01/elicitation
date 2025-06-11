from typing import TypedDict, Optional, List, Literal, Any, Tuple
from dataclasses import dataclass, field
from typing import Dict

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

class Pizza(TypedDict, total=False):
    crust: Optional[PizzaCrust]
    toppings: Optional[List[PizzaTopping]]
    size: Optional[PizzaSize]

class Message(TypedDict):
    role: Literal['caller', 'receiver']
    content: str

@dataclass
class PizzaState:
    pizzas: List[Pizza] = field(default_factory=list)
    conversation: List[Message] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    ambiguous: List[Tuple[int, str]] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

def create_initial_state(pizzas: List[Pizza], rejected: Optional[List[str]] = None, ambiguous: Optional[List[Tuple[int, str]]] = None, errors: Optional[List[str]] = None) -> PizzaState:
    pizzas_with_cheese = []
    for pizza in pizzas:
        toppings = pizza.get('toppings')
        if toppings is None:
            pizza['toppings'] = ['cheese']
        elif 'cheese' not in toppings:
            pizza['toppings'] = toppings + ['cheese']
        pizzas_with_cheese.append(pizza)
    if rejected is None:
        rejected = []
    if ambiguous is None:
        ambiguous = []
    if errors is None:
        errors = []
    return PizzaState(
        pizzas=pizzas_with_cheese,
        conversation=[],
        rejected=rejected,
        ambiguous=ambiguous,
        errors=errors
    )
