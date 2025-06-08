from google import genai
from google.genai import types

client = genai.Client(api_key='AIzaSyA1uZ9JjtPojdpl79y5SRyDq9o3WYqPKuU')

system_prompt = """
You are an AI assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10 . Fun fact - You can aslo do 10 * 3 , which gives the same result

Input: Why is sky blue ?
Output: You alright mate ? Dont know what maths queries are ?
"""

response = client.models.generate_content(
    model = 'gemini-2.0-flash-001' , 
    contents = "Why is sky blue ?",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1
    )
)

print(response.text)