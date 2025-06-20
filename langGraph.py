from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = wrap_openai(OpenAI())

class detectCallResponse(BaseModel):
    is_coding : bool

class State(TypedDict):
    user_message : str
    ai_message : str
    is_coding : bool

def detect_query(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT="""
        You are an AI agent whose job is detect whether the user query is a coding related query or not.
        Return the response in specified JSON format only.
    """

    result = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        response_format=detectCallResponse,
        messages=[
            {"role":"system" , "content":SYSTEM_PROMPT},
            {"role":"user" , "content":user_message}
        ]
    )
    state["is_coding"] = result.choices[0].message.parsed.is_coding
    return state

def route_edge(state:State) -> Literal["solve_coding_ques","solve_simple_ques"]:
    is_coding_question = state.get("is_coding")
    if is_coding_question:
        return "solve_coding_ques"
    else:
        return "solve_simple_ques"

def solve_coding_ques(state:State):
    SYSTEM_PROMPT="""
        You are an AI agent whose job is answer user's coding related query.
        Tone: Professional
    """

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system" , "content":SYSTEM_PROMPT},
            {"role":"user" , "content":state["user_message"]}
        ]
    )
    state["ai_message"] = result.choices[0].message.content
    return state

def solve_simple_question(state:State):
    user_message = state.get("user_message")
    state["ai_message"] = "Please ask some coding related questions."
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

def call_graph(user_query):
    state = {
        "user_message" : user_query,
        "ai_message" : "",
        "is_coding" : False
    }
    result = graph.invoke(state)
    print("Final result : " , result)


while(True):
    user_query = input("> ")
    if(user_query=="Quit" or user_query=="quit"):
        break
    call_graph(user_query)