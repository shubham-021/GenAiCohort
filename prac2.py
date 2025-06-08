import json
from google import genai
from google.genai import types


client = genai.Client(api_key='AIzaSyA1uZ9JjtPojdpl79y5SRyDq9o3WYqPKuU')

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are , you get a user input , you analyse , you think , you again think for several times and then return the output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is , "analyse" , "think" , "output" , "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output Schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step : "string" , content: "string" }}

Example:
Input: What is 2+2 ?
Output: {{ step : "analyse" , content: "Alright! The user is interested in maths query and he is asking a basic arithmetic operation" }}
Output: {{ step : "think" , content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step : "output" , content: "4" }}
Output: {{ step : "validate" , content: "seems like 4 is correct ans for 2+2" }}
Output: {{ step : "result" , content: "2 + 2 = 4 and that is calculated by adding all numbers" }}
"""

response = client.models.generate_content(
    model = 'gemini-2.0-flash-001' , 
    contents =[
        types.Content(
        role = 'user',
        parts = [types.Part.from_text(text = 'What is 2*3+4 ?')]
        ),
        types.Content(
            role = 'model',
            parts = [types.Part.from_text(text = json.dumps({"step": "analyse", "content": "The user wants to evaluate an arithmetic expression involving multiplication and addition."}))]
        ),
        types.Content(
            role = 'model',
            parts = [types.Part.from_text(text = json.dumps({"step": "think", "content": "I need to follow the order of operations (PEMDAS/BODMAS), which means multiplication should be performed before addition."}))]
        ),
        types.Content(
            role = 'model',
            parts = [types.Part.from_text(text = json.dumps({"step": "output", "content": "10"}))]
        ),
        types.Content(
            role = 'model',
            parts = [types.Part.from_text(text = json.dumps({"step": "validate", "content": "2 * 3 = 6, and 6 + 4 = 10. The order of operations has been correctly applied."}))]
        )
    ],
    config = types.GenerateContentConfig(
        system_instruction = system_prompt,
        response_mime_type = 'application/json'
    )
)

print(response.text)