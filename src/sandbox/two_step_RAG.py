import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from settings import settings

def load_documents(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def split_documents(docs, chunk_size=100, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs) 
    return chunks

def create_vector_store(chunks, persist_derictory="./db/chroma_db"):
    embeddings_model = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_derictory=persist_derictory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vector_store

def main():
    URL = "https://docs.langchain.com/oss/python/langchain/rag#rag-chains"
    persist_derictory = "./db/chroma_db"
    if os.path.exists(persist_derictory):
        embeddings_model = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        vector_store = Chroma(
            persist_derictory=persist_derictory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )   
        return vectorstore
    
    documents = load_documents(URL)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, persist_derictory)
    return vector_store

print(main())

