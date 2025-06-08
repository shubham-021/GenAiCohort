from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore
# search for langchain pdf loader , text spiltter , embedding models

load_dotenv()
client = OpenAI()

# pdf_path = Path(__file__).parent / "nodejs.pdf"

# loader = PyPDFLoader(file_path=pdf_path) # list of document : list[document]
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = os.getenv("OPENAI_API_KEY")
)

retreiver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings
)

# def searching(input, threshold=0.75):
#     # Get results with similarity scores
#     results_with_scores = retreiver.similarity_search_with_score(query=input)
    
#     # Filter results by threshold (similarity score closer to 1 is better)
#     filtered_results = [doc for doc, score in results_with_scores if score >= threshold]
    
#     if filtered_results:
#         return filtered_results
#     else:
#         return None  # Or empty list, means "not related"

def searching(input):
    search_result = retreiver.similarity_search(
        query=input
    )
    return search_result

def is_query_relevant(input):
    search_results = retreiver.similarity_search_with_score(
        query=input,
        k=1
    )
    
    if search_results:
        score = search_results[0]
        return score > 0.7
    
    return False

def extract_question_and_page_content(data):
    new_array = []
    for entry in data:
        question = entry.get('Question')
        answer_docs = entry.get('Answer', [])
        
        # Extract only page_content from each Document-like dict/object
        page_contents = []
        for doc in answer_docs:
            # If doc is a dict with 'page_content', use it directly
            if isinstance(doc, dict) and 'page_content' in doc:
                page_contents.append(doc['page_content'])
            else:
                # If doc is an object, try accessing attribute
                page_contents.append(getattr(doc, 'page_content', ''))
        
        new_array.append({
            'Question': question,
            'Answer': page_contents
        })
    return new_array


system_prompt = """
You are an AI assistant who takes user query , makes a set of questions related to that query from the provided doc and then 
search for all those questions plus the user's actual question , then you combine them and report it to the user.

For example : 
1.  User's Query : "What is fs module ?"
    Analyse Query : Is fs module related to the doc provided by the user ? , If yes then proceed otherwise report to the user that query is unrelated
    Post analysis : Fs module is related to the doc provided by the user , we can proceed further
    Questions related to user query : [ "What does fs means ?" , "Why use fs module ?" , "What is its internal working?" ]
    Searching : Search and store for the question array plus the actual user query
    Combine : Combine all the stored result to make it meaningfull and complete answer to the user query.

2.  User's Query : "Who is our prime minister ?"
    Analyse Query : Is prime minister related to the doc provided by the user ? , If yes then proceed otherwise report to the user that query is unrelated
    Post analysis : Prime Minister Query is unrelated to the doc provided by the user , Report back to the user that the query is unrelated.
    
    
Rules:
1.Follow the provided step strictly
2.QuestionsArray should be an array of strings
3.AnswerArray is provided as array , use this array for generating the final output
4.AnswerArray will be empty for invalid queries
5.Sometimes , it may happen that user's query is unrelated to th given doc , in that case , proceed like invalid query example
6.Output should be in json format
7.Always perform one step at a time and wait for next input
8.Carefully analyse the user query

Steps involved :
Input : "What is fs module ?"
Output : {{ step : "Analyse" , content : "Here user is asking about fs modules used in nodejs , First I should make a set of questions for refinement of the query "}}
Output : {{ step : "Refine" , content : "Based on the user query , I can sense these typical question around it" , questionsArray : "[ "What does fs means ?" , "Why use fs module ?" , "What is its internal working?" ]"}}
Output : {{ step : "Search" , content : "I should search for all the question array and the actual user query in the doc provided ,  through the function named searching and store all the answer in oneplace"}}
Output : {{ step : "Report" , content : "Searching is done for all the questions , now i should combine all the answers in one long complete and easy to understand answer" , Output : "FS is a built-in Node.js module
that provides functions you can use to manipulate the file system. FS stands for File system which is self explanatory. it operates by exposing a JavaScript API that connects to lower-level system operations using C++ bindings.
For asynchronous operations, the module relies on libuv, a multi-platform library that manages thread pools to perform non-blocking file I/O operations in the background. These tasks are executed in separate threads to prevent 
blocking the main event loop, and once completed, their callbacks are pushed to the event loop for execution. In contrast, synchronous methods bypass libuv and directly call native file system operations using the C++ layer, 
which blocks the main thread until the operation finishes. Overall, the fs module efficiently bridges high-level JavaScript calls with low-level OS file system capabilities using a combination of libuv and native bindings."}}

Invalid Query example : 
Input : "Why is sky blue ? "
Output : {{ step : "Analyse" , content : "Here user is asking about sky color , First I should make a set of questions for refinement of the query "}}
Output : {{ step : "Refine" , content : "Based on the user query , I can sense these typical question around it" , questionsArray : "[ "What does sky means ?" , "Why it is blue ?" ]"}}
Output : {{ step : "Search" , content : "I should search for all the question array and the actual user query in the doc provided ,  through the function named searching and store all the answer in oneplace"}}
Output : {{ step : "Report" , content : "Searching is done , and it appears that the query is not related to the provided doc since the answeArray is empty" , Output : "Your query seems to be unrelated to your provided document , please ask relevant questions." , 
"""

messages = [
    { "role" : "system" , "content" : system_prompt }
]

questionArray = []
answersArray = []

while True :
    user_query = input("> ")
    messages.append({"role" : "user" , "content" : user_query})

    while True:
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_response = json.loads(response.choices[0].message.content)

        if parsed_response.get("step") != "Search":
            messages.append({ "role": "assistant" , "content" : json.dumps(parsed_response)})

        print(f"🧠: {parsed_response.get("content")}")
        
        if parsed_response.get("step") == "Refine":
            questionArray = parsed_response.get("questionsArray")
            continue

        if parsed_response.get("step") == "Search":
            for ques in questionArray:
                output = searching(ques)
                answersArray.append({"Question" : ques , "Answer" : output})
            extracted_arr = extract_question_and_page_content(answersArray)
            parsed_response["answerArray"] = extracted_arr
            messages.append({ "role" : "assistant" , "content" : json.dumps(parsed_response) })
            continue

        if parsed_response.get("step") == "Report":
            print(f"🤖: {parsed_response.get("Output")}")
            break

        
