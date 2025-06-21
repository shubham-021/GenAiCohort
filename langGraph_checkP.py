from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model_provider="openai" , model="gpt-4o")

class State(TypedDict):
    messages : Annotated[list , add_messages]

def chatbot(state:State):
    return {"messages" : llm.invoke(state["messages"])}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot" , chatbot)

graph_builder.add_edge(START , "chatbot")
graph_builder.add_edge("chatbot" , END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    intial_state = {"messages" : [{"role" : "user" , "content" : user_input}]}
    # result = graph.invoke(intial_state)
    # aiMessage = [msg.content for msg in result['messages'] if isinstance(msg,AIMessage)]
    # for msg in aiMessage:
    #     print(msg)
    for event in graph.stream(intial_state):
        for value in event.values():
            print("Assistant:", value["messages"].content)
    # for event in graph.stream(intial_state , stream_mode="values"):
    #     if "messages" in event:
    #         event["messages"][-1].pretty_print()


def chat():
    user_query = input("You : ")
    stream_graph_updates(user_input=user_query)

chat()