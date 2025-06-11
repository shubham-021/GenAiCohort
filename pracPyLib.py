import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.schema.output import ChatGeneration
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openAI_client = OpenAI()
qdrant_client = QdrantClient(url="http://localhost:6333")
BASE_URL = "https://docs.chaicode.com/"
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = os.getenv("OPENAI_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

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
#         formatted_course_name = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", course_name.lower()))

#         course_links = {}
#         for link in details.select('ul a[href]'):
#             page_name = link.get_text(strip=True)
#             page_url = f"{BASE_URL}{link['href']}"
#             course_links[page_name] = page_url

#         course_structure[formatted_course_name] = course_links


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
# # }

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
                },
                "user_query" : {
                    "type": "string",
                    "description": "users query"
                }
            },
            "required": ["collection_name" , "user_query"]
        }
    }
}

llm1 = ChatOpenAI(model="gpt-4")
llm2 = ChatOpenAI(model="gpt-4")

def similarity_search(course_name , user_query):
    retreiver = QdrantVectorStore.from_existing_collection(
                    url="http://localhost:6333",
                    collection_name=course_name,
                    embedding=embeddings
                )
    search_result = retreiver.similarity_search(
        query=user_query
    )
    return search_result

prompt1 = PromptTemplate.from_template("""
    You are an AI assistant which takes user query , analyse , and finds the relevant course name from the available courses and
    strictly returns only the course name as per available course lists. If user query is unrelated return "None".

    User's Query : {query}
    Courses Available : {courseLists}
""")

prompt2 = PromptTemplate.from_template("""
    You are an AI assistant which takes chunks retrieved from the db and the user query , and gives a nice and clear reponse to the user , along with the course name
    and source from the source list.If you get "none" as context , tell user to give his query more clearly , as we can find no related 
    document on chai aur docs.
                                       
    user's query : {query}
    Chunk : {context}  
    course name : {course_name}   
    sources : {sources}                          
""")

users_query = input("> ")

course_names = ["Chai aur C++" , "Chai aur DevOps" , "Chai aur Django" , "Chai aur SQL" , "Chai aur HTML" , "Chai aur Git"]

chain1 = prompt1 | llm1
course_name_response = chain1.invoke({
    "query" : users_query,
    "courseLists" : course_names
})

predicted_course = course_name_response.content.strip()
formatted_course = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", predicted_course.lower()))
# print(formatted_course)
if formatted_course == "none":
        response = (prompt2 | llm2).invoke({
        "query": users_query,
        "context": formatted_course,
        "course_name": predicted_course
    })
elif formatted_course != "none":
    docs = similarity_search(formatted_course, users_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set(doc.metadata["source"] for doc in docs))
    # print(docs)
    response = (prompt2 | llm2).invoke({
        "query": users_query,
        "context": context,
        "course_name": predicted_course,
        "sources" : sources
    })

print(response.content)


