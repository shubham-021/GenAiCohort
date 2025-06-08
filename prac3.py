from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()

client = OpenAI()

def get_weather(city: str):
    return "31 degree celcius"

def run_command(command):
    result = os.system(command=command)
    return result

available_tools = {
    "get_weather" : {
        "fn" : get_weather,
        "description" : "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command" : {
        "fn" : run_command,
        "description" : "Takes a command from the user and executes it"
    }
}

system_prompt = """
    You are an helpful AI assistant who is specialized in resolving user queries.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    1. Follow the strict JSON output as per Output Schema.
    2. Always perform one step at a time and wait for next input
    3. Carefully analyse the user query

    Output JSON format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function"
    }}

    Available Tools:
    - get_weather: Takes a city name as input and returns the current weather for the city
    - run_command: Takes a command from the user and executes it

    Example:
    User query: What is the weather of new york ?
    Output: {{ "step" : "plan" , "content": "The user is interested in weather data of new york" }}
    Output: {{ "step" : "plan" , "content": "From the available tools I should call get_weather" }}
    Output: {{ "step" : "action" , "function": "get_weather" , "input" : "new york" }}
    Output: {{ "step" : "observe" , "output": "12 Degree Cel" }}
    Output: {{ "step" : "output" , "content": "The weather for new york seems to be 12 degrees" }}
    
"""

messages = [
    { "role" : "system" , "content" : system_prompt }
]

while True:
    user_query = input('> ')
    messages.append({ "role": "user" , "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_response = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant" , "content" : json.dumps(parsed_response)})

        if parsed_response.get("step") == "plan":
            print(f"🧠: {parsed_response.get("content")}")
            continue

        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            if available_tools.get(tool_name , False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append({"role": "assistant" , "content": json.dumps({"steps": "observe" , "output" : output})})
                continue
        
        if parsed_response.get("step") == "output":
            print(f"🤖: {parsed_response.get("content")}")
            break

    

