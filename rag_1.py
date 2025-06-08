from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore
# search for langchain pdf loader , text spiltter , embedding models

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(file_path=pdf_path) # list of document : list[document]
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = os.getenv("OPENAI_API_KEY")
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings
# )

# vector_store.add_documents(documents = split_docs)
# print("Injection Done")

retreiver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings
)

search_result = retreiver.similarity_search(
    query="What is FS Module ?"
)

print("Relevant Chunks : " , search_result)



