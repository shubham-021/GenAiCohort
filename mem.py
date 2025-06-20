from mem0 import Memory
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

config = {
    "version" : "v1:1",
    "embeder" : {
        "provider" : "openai",
        "config" : {"api_key" : os.getenv("OPENAI_API_KEY") , "model" : "text-embedding-3-small"}
    }, 
    "llm" : {"provider" : "openai" , "config" : {"api_key" : os.getenv("OPENAI_API_KEY") , "model" : "gpt-4.1"}},
    "vector_store" : {
        "provider" : "qdrant",
        "config" : {
            "host" : "localhost",
            "port" : "6333"
        }
    },
    "graph_store" : {
        "provider" : "neo4j",
        "config" : {
            "url" : "bolt://localhost:7687" , "username" : os.getenv("NEO4J_USERNAME") , "password" : os.getenv("NEO4J_PASSWORD")
        }
    }
}

mem_client = Memory.from_config(config)
openAI_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(message):
    mem_result = mem_client.search(query=message , user_id="s2100")

    memories = "\n".join([f"{m['memory']} {m['score']}" for m in mem_result.get("results")])
    # print(memories) 

    SYSTEM_PROMPT = f"""
    You are a memory-aware conversational assistant , you chat with the user with appropriate response to their queries , and if they want any fact
    checking you use your provided memory context to generate response. 

    Tone : Friendly

    Memory and Score: 
    {memories}
    """

    messages = [
        {"role" : "system" , "content" : SYSTEM_PROMPT },
        {"role" : "user" , "content" : message}
    ]

    result = openAI_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )

    messages.append(
        {"role" : "assistant" , "content" : result.choices[0].message.content}
    )

    mem_client.add(messages , user_id="s2100")

    return result.choices[0].message.content

while(True):
    user_message = input("You : ")
    print("GPT : " , chat(message=user_message))