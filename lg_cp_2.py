from lg_cp import create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver

DB_URI = "mongodb://admin:admin@localhost:27017"
config = { "configurable":{"thread_id":"1"} }

def init():
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        gaph_with_mongo = create_chat_graph(checkpointer=checkpointer)

        while True:
            user_input = input("You : ")
            intial_state = {"messages" : [{"role" : "user" , "content" : user_input}]}
            for event in gaph_with_mongo.stream(intial_state , config ,  stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()


init()