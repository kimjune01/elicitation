from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import graph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(graph, CompiledStateGraph)

