from human_in_the_loop import create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
import json

DB_URI = "mongodb://admin:admin@localhost:27017"
config = { "configurable":{"thread_id":"12"} }

def init():
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)

        state = graph_with_mongo.get_state(config=config)
        # for message in state.values["messages"]:
        #     message.pretty_print()

        last_message = state.values["messages"][-1]
        tools_calls = last_message.additional_kwargs.get("tool_calls",[])

        user_query = None

        for call in tools_calls:
            if call.get("function" , {}).get("name") == "human_assistance" :
                args = call["function"].get("arguments" , {})
                try:
                    args_dict = json.loads(args)
                    user_query = args_dict.get("query")
                except json.JSONDecodeError:
                    print("Failed to decode function arguments")
                    
        print("Query : " , user_query)
        ans = input("Admin Response : ")
        human_command = Command(resume={"data": ans})
        for event in graph_with_mongo.stream(human_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print() 



init()