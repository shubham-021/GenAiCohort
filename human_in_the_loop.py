from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages : Annotated[list , add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str):
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

llm = init_chat_model(model_provider="openai" , model="gpt-4o")
tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}



graph_builder.add_node("chatbot" , chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START , "chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot" , END)

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)