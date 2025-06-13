import pytest
from unittest.mock import patch, MagicMock
from src.agent import nodes
from src.agent.state import PizzaState, create_initial_state, Pizza
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def basic_state():
    return PizzaState(
        pizzas=[Pizza(crust='thin', toppings=['cheese'], size='small')],
        messages=[HumanMessage(content='I want a small thin pizza with cheese.')],
        rejected=[],
        ambiguous=[],
        questions=[],
        errors=[]
    )

@pytest.fixture
def empty_state():
    return PizzaState(
        pizzas=[],
        messages=[],
        rejected=[],
        ambiguous=[],
        questions=[],
        errors=[]
    )

@pytest.fixture
def incomplete_pizza_state():
    return PizzaState(
        pizzas=[Pizza(crust=None, toppings=['cheese'], size=None)],
        messages=[HumanMessage(content='I want a pizza with cheese.')],
        rejected=[],
        ambiguous=[(0, 'crust'), (0, 'size')],
        questions=[],
        errors=[]
    )

@pytest.fixture
def multiple_pizza_state():
    return PizzaState(
        pizzas=[
            Pizza(crust='thin', toppings=['cheese'], size='small'),
            Pizza(crust='stuffed', toppings=['pepperoni'], size='large')
        ],
        messages=[HumanMessage(content='I want two pizzas.')],
        rejected=[],
        ambiguous=[],
        questions=[],
        errors=[]
    )

# Test validate_pizzas function
def test_validate_pizzas_complete_pizza(basic_state):
    """Test validate_pizzas with a complete pizza"""
    state = nodes.validate_pizzas(basic_state)
    assert isinstance(state, PizzaState)
    assert hasattr(state, 'questions')
    assert len(state.questions) == 0  # Complete pizza should have no questions

def test_validate_pizzas_incomplete_pizza(incomplete_pizza_state):
    """Test validate_pizzas with an incomplete pizza"""
    state = nodes.validate_pizzas(incomplete_pizza_state)
    assert isinstance(state, PizzaState)
    assert len(state.questions) == 2  # Missing crust and size
    assert "What crust for pizza #1?" in state.questions
    assert "What size for pizza #1?" in state.questions

def test_validate_pizzas_empty_state(empty_state):
    """Test validate_pizzas with no pizzas"""
    state = nodes.validate_pizzas(empty_state)
    assert isinstance(state, PizzaState)
    assert len(state.questions) == 0

# Test generate_ambiguities function
def test_generate_ambiguities_complete_pizza(basic_state):
    """Test generate_ambiguities with complete pizza"""
    ambiguities = nodes.generate_ambiguities(basic_state)
    assert isinstance(ambiguities, list)
    assert len(ambiguities) == 0

def test_generate_ambiguities_incomplete_pizza():
    """Test generate_ambiguities with incomplete pizza"""
    state = PizzaState(
        pizzas=[Pizza(crust=None, toppings=None, size='small')],
        messages=[],
        rejected=[], ambiguous=[], questions=[], errors=[]
    )
    ambiguities = nodes.generate_ambiguities(state)
    assert len(ambiguities) == 2
    assert (0, 'crust') in ambiguities
    assert (0, 'toppings') in ambiguities

def test_generate_ambiguities_empty_state(empty_state):
    """Test generate_ambiguities with no pizzas"""
    ambiguities = nodes.generate_ambiguities(empty_state)
    assert isinstance(ambiguities, list)
    assert len(ambiguities) == 0

# Test format_messages function
def test_format_messages_human_ai():
    """Test format_messages with HumanMessage and AIMessage"""
    messages = [
        HumanMessage(content="I want a pizza"),
        AIMessage(content="What size would you like?")
    ]
    result = nodes.format_messages(messages)
    expected = "human: I want a pizza\nai: What size would you like?"
    assert result == expected

def test_format_messages_dict_format():
    """Test format_messages with legacy dict format"""
    messages = [
        {"role": "caller", "content": "I want a pizza"},
        {"role": "receiver", "content": "What size?"}
    ]
    result = nodes.format_messages(messages)
    expected = "caller: I want a pizza\nreceiver: What size?"
    assert result == expected

def test_format_messages_empty():
    """Test format_messages with empty list"""
    result = nodes.format_messages([])
    assert result == ""

# Test build_pizza_extraction_prompt function
def test_build_pizza_extraction_prompt():
    """Test build_pizza_extraction_prompt"""
    messages = [HumanMessage(content="I want a large pepperoni pizza")]
    result = nodes.build_pizza_extraction_prompt(messages)
    assert isinstance(result, str)
    assert "human: I want a large pepperoni pizza" in result

# Test parse_llm_pizza_response function
def test_parse_llm_pizza_response_valid_json():
    """Test parse_llm_pizza_response with valid JSON"""
    response = '{"pizzas": [{"crust": "thin", "toppings": ["cheese"], "size": "small"}], "rejected": [], "ambiguous": []}'
    pizzas, rejected, ambiguous, errors = nodes.parse_llm_pizza_response(response)
    assert len(pizzas) == 1
    assert pizzas[0]["crust"] == "thin"
    assert len(rejected) == 0
    assert len(ambiguous) == 0
    assert len(errors) == 0

def test_parse_llm_pizza_response_code_block():
    """Test parse_llm_pizza_response with JSON in code block"""
    response = '```json\n{"pizzas": [{"crust": "thin", "toppings": ["cheese"], "size": "small"}], "rejected": [], "ambiguous": []}\n```'
    pizzas, rejected, ambiguous, errors = nodes.parse_llm_pizza_response(response)
    assert len(pizzas) == 1
    assert len(errors) == 0

def test_parse_llm_pizza_response_invalid_json():
    """Test parse_llm_pizza_response with invalid JSON"""
    response = 'invalid json response'
    pizzas, rejected, ambiguous, errors = nodes.parse_llm_pizza_response(response)
    assert len(pizzas) == 0
    assert len(errors) > 0
    assert "Failed to parse LLM response as JSON" in errors[0]

def test_parse_llm_pizza_response_empty_fields():
    """Test parse_llm_pizza_response with empty string fields"""
    response = '{"pizzas": [{"crust": "", "toppings": ["cheese"], "size": ""}], "rejected": [], "ambiguous": []}'
    pizzas, rejected, ambiguous, errors = nodes.parse_llm_pizza_response(response)
    assert len(pizzas) == 1
    assert pizzas[0]["crust"] is None
    assert pizzas[0]["size"] is None

# Test inspect_state_node function
def test_inspect_state_node_basic(basic_state, capsys):
    """Test inspect_state_node with basic state"""
    state = nodes.inspect_state_node(basic_state)
    assert state is basic_state
    captured = capsys.readouterr()
    assert "INSPECT STATE" in captured.out

def test_inspect_state_node_with_errors(capsys):
    """Test inspect_state_node with errors in state"""
    state = PizzaState(
        pizzas=[], messages=[], rejected=[], ambiguous=[], questions=[],
        errors=["Raw response: some error response"]
    )
    result = nodes.inspect_state_node(state)
    assert result is state
    captured = capsys.readouterr()
    assert "RAW LLM OUTPUT" in captured.out

# Test compute_pizza_completeness function
def test_compute_pizza_completeness_complete_pizzas(basic_state):
    """Test compute_pizza_completeness with complete pizzas"""
    complete, incomplete = nodes.compute_pizza_completeness(basic_state)
    assert isinstance(complete, list)
    assert isinstance(incomplete, list)
    assert len(complete) == 1
    assert len(incomplete) == 0

def test_compute_pizza_completeness_incomplete_pizzas(incomplete_pizza_state):
    """Test compute_pizza_completeness with incomplete pizzas"""
    complete, incomplete = nodes.compute_pizza_completeness(incomplete_pizza_state)
    assert len(complete) == 0
    assert len(incomplete) == 1
    assert incomplete[0]['index'] == 'first'
    assert 'crust' in incomplete[0]['missing_fields']
    assert 'size' in incomplete[0]['missing_fields']

def test_compute_pizza_completeness_mixed_pizzas():
    """Test compute_pizza_completeness with mix of complete and incomplete pizzas"""
    state = PizzaState(
        pizzas=[
            Pizza(crust='classic', toppings=['cheese'], size='medium'),  # Complete
            Pizza(crust=None, toppings=['pepperoni'], size=None)         # Incomplete
        ],
        messages=[], rejected=[], ambiguous=[], questions=[], errors=[]
    )
    complete, incomplete = nodes.compute_pizza_completeness(state)
    assert len(complete) == 1
    assert len(incomplete) == 1

# Test ordinal function
def test_ordinal_function():
    """Test ordinal number conversion"""
    assert nodes.ordinal(1) == "first"
    assert nodes.ordinal(2) == "second"
    assert nodes.ordinal(3) == "third"
    assert nodes.ordinal(21) == "21th"

# Test make_accepted_fields function
def test_make_accepted_fields():
    """Test make_accepted_fields function"""
    incomplete_pizzas = [
        {
            'pizza': Pizza(crust='thin', toppings=['cheese'], size=None),
            'missing_fields': ['size']
        }
    ]
    result = nodes.make_accepted_fields(incomplete_pizzas)
    assert len(result) == 1
    assert result[0]['crust'] == 'thin'
    assert result[0]['toppings'] == ['cheese']
    assert 'size' not in result[0]

# Test extract_pizzas_node function
@patch('src.agent.nodes.gemini_llm')
def test_extract_pizzas_node_success(mock_llm, basic_state):
    """Test extract_pizzas_node with successful LLM response"""
    mock_llm.return_value = '{"pizzas": [{"crust": "thin", "toppings": ["cheese"], "size": "small"}], "rejected": [], "ambiguous": []}'
    new_state = nodes.extract_pizzas_node(basic_state)
    assert isinstance(new_state, PizzaState)
    assert len(new_state.pizzas) == 1
    assert new_state.messages == basic_state.messages

@patch('src.agent.nodes.gemini_llm')
def test_extract_pizzas_node_llm_failure(mock_llm, basic_state):
    """Test extract_pizzas_node with LLM failure"""
    mock_llm.side_effect = Exception("API Error")
    new_state = nodes.extract_pizzas_node(basic_state)
    assert isinstance(new_state, PizzaState)
    assert len(new_state.pizzas) == 0
    assert len(new_state.errors) > 0
    assert "LLM call failed" in new_state.errors[0]

# Test elicitation_response_node function
@patch('src.agent.nodes.gemini_llm')
def test_elicitation_response_node_success(mock_llm, incomplete_pizza_state):
    """Test elicitation_response_node with successful response"""
    mock_llm.return_value = 'What crust and size would you like?'
    original_message_count = len(incomplete_pizza_state.messages)
    state = nodes.elicitation_response_node(incomplete_pizza_state)
    
    assert isinstance(state, PizzaState)
    assert len(state.messages) == original_message_count + 1
    assert isinstance(state.messages[-1], AIMessage)
    assert state.messages[-1].content == 'What crust and size would you like?'

# Test order_confirmation_node function
def test_order_confirmation_node_success(basic_state):
    """Test order_confirmation_node with complete pizzas"""
    original_message_count = len(basic_state.messages)
    state = nodes.order_confirmation_node(basic_state)
    
    assert isinstance(state, PizzaState)
    assert len(state.messages) == original_message_count + 1
    assert isinstance(state.messages[-1], AIMessage)
    assert "Your pizza order is complete!" in state.messages[-1].content

def test_order_confirmation_node_with_incomplete_pizzas(incomplete_pizza_state):
    """Test order_confirmation_node with incomplete pizzas (should fail)"""
    with pytest.raises(AssertionError, match="order_confirmation_node called with incomplete pizzas"):
        nodes.order_confirmation_node(incomplete_pizza_state)

def test_order_confirmation_node_multiple_pizzas(multiple_pizza_state):
    """Test order_confirmation_node with multiple complete pizzas"""
    state = nodes.order_confirmation_node(multiple_pizza_state)
    confirmation_content = state.messages[-1].content
    
    assert "first pizza" in confirmation_content
    assert "second pizza" in confirmation_content
    assert "small thin crust" in confirmation_content
    assert "large stuffed crust" in confirmation_content

# Test human_node function
def test_human_node(basic_state):
    """Test human_node function"""
    result = nodes.human_node(basic_state)
    assert result is basic_state
    assert isinstance(result, PizzaState)
