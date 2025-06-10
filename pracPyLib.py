import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import Tool
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openAI_client = OpenAI()
# qdrant_client = QdrantClient(url="http://localhost:6333")
# BASE_URL = "https://docs.chaicode.com/"
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     api_key = os.getenv("OPENAI_API_KEY")
# )
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200,
# )

# vector_stores = {}
# course_structure = {}
# course_docs = {}

# r = requests.get('https://docs.chaicode.com/youtube/getting-started/')
# soup = BeautifulSoup(r.text, 'html.parser')
# sidebar = soup.find('nav' , {'aria-label' : 'Main'})
# top_level_items = sidebar.select('.top-level > li')



# for item in top_level_items:
#     details = item.find('details')
#     if details:
#         course_name = details.find('span' , class_='large').get_text(strip=True)

#         course_links = {}
#         for link in details.select('ul a[href]'):
#             page_name = link.get_text(strip=True)
#             page_url = f"{BASE_URL}{link['href']}"
#             course_links[page_name] = page_url

#         course_structure[course_name] = course_links


# for course_name in course_structure.keys():
#     try:
#         qdrant_client.get_collection(course_name)
#     except:
#         qdrant_client.create_collection(
#             collection_name=course_name,
#             vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
#         )
    
#     vector_stores[course_name] = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name=course_name,
#         embedding=embeddings
#     )


# for course_name , pages in course_structure.items():
#     print(f"\nProcessing course: {course_name}")
#     course_chunk = []

#     for page_name , page_url in pages.items():
#         print(f"Loading page: {page_name} from {page_url}")
#         loader = WebBaseLoader(page_url)
#         docs = loader.load()

#         for doc in docs:
#             doc.metadata.update({
#                 "source": page_url,
#                 "course": course_name,
#                 "page": page_name
#             })
        
#         chunks = text_splitter.split_documents(docs)
#         course_chunk.extend(chunks)
    
#     course_docs[course_name] = course_chunk


# for course_name , docs in course_docs.items():
#     vector_stores[course_name].add_documents(docs)
#     print(f"Indexed {len(docs)} chunks for course: {course_name}")

# tool = {
#   "name": "similarity_search",
#   "description": "searches for the chunks related to the user query from the collection",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "collection_name": {
#         "type": "string",
#         "description": "Name of the course , the user's query might be related to"
#       },
#     },
#     "required": ["collection_name"]
#   }
# }

tool = {
    "type": "function",
    "function": {
        "name": "similarity_search",
        "description": "Searches for the chunks related to the user query from the collection",
        "parameters": {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the course the user's query might be related to"
                }
            },
            "required": ["collection_name"]
        }
    }
}


system_prompt = """
    You are an AI assistant which answers users query based on the chai aur docs website , you take user query and run a similarity_search
    function given as tool , which takes name of the collection as an argument , which you can pick from the provided available collections
    based on the user query , then you get chunks of datas as response from that function , which you present in front of the user.If you
    get no chunks or get false as a response from the function , ask user to send their query more clearly.

    Collection/Courses are named as "Chai aur"+" course name"

    Available collections are :
    1. Chai aur C++
    2. Chai aur DevOps
    3. Chai aur Django
    4. Chai aur Git
    5. Chai aur HTML
    6. Chai aur SQL

    Steps to be followed:
    1. Take the user query
    2. Find the relevant course name , if no related course then tell user that query is unrelated
    3. If course found , call for similarity_search function with the related collection name as argument
    4. Respond to users query based on the chunks retrieved 
    5. If query is not related to any of the collection/courses , ask user to state their query more clearly
"""

response = openAI_client.chat.completions.create(
  model="gpt-4o",
  messages=[
      {"role" : "system" , "content": system_prompt},
      {"role" : "user" , "content" : "How can you assist me ?"}
    ],
  tools = [tool],
  tool_choice = "auto"
)

tool_calls = response.choices[0].message.tool_calls

if tool_calls:
    tool_name = tool_calls[0].function.name
    tool_args = json.loads(tool_calls[0].function.arguments)
    print("Tool name:", tool_name)
    print("Args:", tool_args)
else:
    print("No tool was called. Model said:")
    print(response.choices[0].message.content)