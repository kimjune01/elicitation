import pytest
from unittest.mock import patch, MagicMock
from src.agent import nodes
from src.agent.state import PizzaState, create_initial_state
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def basic_state():
    from src.agent.state import Pizza
    return PizzaState(
        pizzas=[Pizza(crust='thin', toppings=['cheese'], size='small')],
        messages=[HumanMessage(content='I want a small thin pizza with cheese.')],
        rejected=[],
        ambiguous=[],
        questions=[],
        errors=[]
    )

def test_validate_pizzas(basic_state):
    state = nodes.validate_pizzas(basic_state)
    assert isinstance(state, PizzaState)
    assert hasattr(state, 'questions')

def test_generate_ambiguities(basic_state):
    ambiguities = nodes.generate_ambiguities(basic_state)
    assert isinstance(ambiguities, list)

def test_inspect_state_node(basic_state, capsys):
    state = nodes.inspect_state_node(basic_state)
    assert state is basic_state
    captured = capsys.readouterr()
    assert "INSPECT STATE" in captured.out

def test_compute_pizza_completeness(basic_state):
    complete, incomplete = nodes.compute_pizza_completeness(basic_state)
    assert isinstance(complete, list)
    assert isinstance(incomplete, list)

@patch('src.agent.nodes.gemini_llm')
def test_extract_pizzas_node(mock_llm, basic_state):
    mock_llm.return_value = '{"pizzas": [{"crust": "thin", "toppings": ["cheese"], "size": "small"}], "rejected": [], "ambiguous": []}'
    new_state = nodes.extract_pizzas_node(basic_state, mock_llm)
    assert isinstance(new_state, PizzaState)
    assert new_state.pizzas

@patch('src.agent.nodes.gemini_llm')
def test_elicitation_response_node(mock_llm, basic_state):
    mock_llm.return_value = 'Please specify the size.'
    state = nodes.elicitation_response_node(basic_state)
    assert isinstance(state, PizzaState)
    assert isinstance(state.messages[-1], AIMessage)

def test_order_confirmation_node(basic_state):
    # Make sure there are no incomplete pizzas
    from src.agent.state import Pizza
    state = PizzaState(
        pizzas=[Pizza(crust='thin', toppings=['cheese'], size='small')],
        messages=[],
        questions=[],
        rejected=[],
        ambiguous=[],
        errors=[],
    )
    state = nodes.order_confirmation_node(state)
    assert state.questions
