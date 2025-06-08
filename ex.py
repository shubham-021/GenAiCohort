import json
response = '{ "step" : "Search" , "content" : "I should search for all the question array and the actual user query in the doc provided ,  through the function named searching and store all the answer in oneplace"}'
jsonResponse = json.loads(response)

answerArray = ["My name is Shubham"]
jsonResponse["asnwerArray"] = answerArray

print(jsonResponse)
print(type(jsonResponse))