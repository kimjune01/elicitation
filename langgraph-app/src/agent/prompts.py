from langchain.prompts import PromptTemplate


PIZZA_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["conversation"],
    template="""
You are a pizza order extractor.
Here is the conversation so far:
{conversation}
Extract up to 100 pizzas from the conversation with the caller. Each pizza should be a JSON object compatible with the following PizzaState schema:

PizzaState.pizzas is an array of pizza objects, where each pizza has:
- crust: one of [\"thin\", \"classic\", \"stuffed\"]
- toppings: an array (up to 5) of any of the following: [\"pepperoni\", \"mushrooms\", \"onions\", \"sausage\", \"bacon\", \"extra cheese\", \"black olives\", \"green peppers\", \"pineapple\", \"spinach\", \"ham\", \"tomatoes\", \"chicken\", \"beef\", \"anchovies\", \"jalapenos\", \"garlic\", \"artichokes\", \"broccoli\", \"feta cheese\", \"salami\", \"red onions\", \"corn\", \"zucchini\", \"eggplant\", \"prosciutto\", \"basil\", \"sun-dried tomatoes\", \"roasted red peppers\", \"arugula\"]
- size: one of [\"small\", \"medium\", \"large\", \"extra_large\"]

Return a JSON object with the following structure (and no other text):
{{
  \"pizzas\": [ ... ], // array of pizza objects as described above
  \"rejected\": [ ... ], // array of strings describing any orders or items that could not be interpreted as a valid pizza
  \"ambiguous\": [ ... ] // array of [pizza_index, field_name] for any pizzas with missing or unclear fields (field_name is one of 'crust', 'toppings', 'size')
}}

Return only plain JSON, without any markdown or code blocks.
"""
)