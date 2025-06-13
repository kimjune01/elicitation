from langchain.prompts import PromptTemplate


PIZZA_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["messages"],
    template="""
You are a pizza order extractor.
Here is the messages so far:
{messages}
Extract up to 100 pizzas from the messages with the caller. Each pizza should be a JSON object compatible with the following PizzaState schema:

PizzaState.pizzas is an array of pizza objects, where each pizza has:
- crust: one of [\"thin\", \"classic\", \"stuffed\"] (if not specified, leave this field empty)
- toppings: an array (up to 5) of any of the following: [\"pepperoni\", \"mushrooms\", \"onions\", \"sausage\", \"bacon\", \"extra cheese\", \"black olives\", \"green peppers\", \"pineapple\", \"spinach\", \"ham\", \"tomatoes\", \"chicken\", \"beef\", \"anchovies\", \"jalapenos\", \"garlic\", \"artichokes\", \"broccoli\", \"feta cheese\", \"salami\", \"red onions\", \"corn\", \"zucchini\", \"eggplant\", \"prosciutto\", \"basil\", \"sun-dried tomatoes\", \"roasted red peppers\", \"arugula\"]
- size: one of [\"small\", \"medium\", \"large\", \"extra_large\"] (if not specified, leave this field empty)

Return a JSON object with the following structure (and no other text):
{{
  \"pizzas\": [ ... ], // array of pizza objects as described above
  \"rejected\": [ ... ], // array of strings describing any orders or items that could not be interpreted as a valid pizza
  \"ambiguous\": [ ... ] // array of [pizza_index, field_name] for any pizzas with missing or unclear fields (field_name is one of 'crust', 'toppings', 'size')
}}

Return only plain JSON, without any markdown or code blocks.
"""
)

ORDER_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["accepted", "rejected", "ambiguous", "missing", "complete_pizzas"],
    template="""
You are an efficient pizza ordering assistant practicing active listening. Your job is to summarize the current pizza order by specifically mentioning what the customer has told you. Acknowledge which parts of a pizza has been accepted individually. Mention what was rejected if any, and what needs clarification. 

--- ORDER SUMMARY ---
Accepted Pizza Properties (acknowledge specifically what was mentioned):
{accepted}

--- REJECTED ITEMS (acknowledge and say that it's not available') ---
{rejected}

--- AMBIGUOUS PIZZA PROPERTIES (need clarification) ---
{ambiguous}

--- MISSING PIZZA PROPERTIES (need details) ---
{missing}

--- COMPLETE PIZZAS (acknowledge and do not ask for clarification) ---
{complete_pizzas}

Each pizza can have the following properties:
- crust: one of [\"thin\", \"classic\", \"stuffed\"] (if not specified, leave this field empty)
- toppings: an array (up to 5) of any of the following: [\"pepperoni\", \"mushrooms\", \"onions\", \"sausage\", \"bacon\", \"extra cheese\", \"black olives\", \"green peppers\", \"pineapple\", \"spinach\", \"ham\", \"tomatoes\", \"chicken\", \"beef\", \"anchovies\", \"jalapenos\", \"garlic\", \"artichokes\", \"broccoli\", \"feta cheese\", \"salami\", \"red onions\", \"corn\", \"zucchini\", \"eggplant\", \"prosciutto\", \"basil\", \"sun-dried tomatoes\", \"roasted red peppers\", \"arugula\"]
- size: one of [\"small\", \"medium\", \"large\", \"extra_large\"] (if not specified, leave this field empty)

IMPORTANT: When acknowledging accepted items, always repeat back the specific details the customer mentioned (e.g., "I have <crust> <toppings> <size> for your first pizza" rather than just "I have toppings for two pizzas"). This shows you're actively listening and helps avoid confusion.
When crust or size is rejected or missing, mention the options available.
When toppings are rejected or missing, simply say that it's not available.

Respond in natural language without headers or delimiters.

"""
)

