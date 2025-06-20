from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    user_message : str
    ai_message : str
    is_coding : bool

def detect_query(state: State):
    user_message = state.get("user_message")
    #openAI call
    state["is_coding"] = True
    return state

def route_edge(state:State) -> Literal["solve_coding_ques","solve_simple_ques"]:
    is_coding_question = state.get("is_coding")
    if is_coding_question:
        return "solve_coding_ques"
    else:
        return "solve_simple_ques"

def solve_coding_ques(state:State):
    state["ai_message"] = "Here is your coding ques ans"
    return state

def solve_simple_question(state:State):
    user_message = state.get("user_message")
    state.ai_message = "Please ask some coding related questions."
    return state


graph_builder = StateGraph(State)

graph_builder.add_node("detect_query" , detect_query)
graph_builder.add_node("solve_coding_ques" , solve_coding_ques)
graph_builder.add_node("solve_simple_ques" , solve_simple_question)
graph_builder.add_node("route_edge" , route_edge )

graph_builder.add_edge(START,"detect_query")
graph_builder.add_conditional_edges("detect_query",route_edge)
graph_builder.add_edge("solve_coding_ques",END)
graph_builder.add_edge("solve_simple_ques",END)

graph = graph_builder.compile()

def call_graph():
    state = {
        "user_message" : "Hi ! How are you ?",
        "ai_message" : "",
        "is_coding" : False
    }
    result = graph.invoke(state)
    print("Final result : " , result)

call_graph()