import os
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.sandbox.settings import settings

URL = "https://docs.langchain.com/oss/python/langchain/rag#rag-chains"
PERSIST_DIRECTORY = "./db/chroma_db"

def load_documents(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    return loader.load()

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

def build_or_load_vectorstore() -> Chroma:
    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    if vector_store._collection.count() == 0:
        print("Creating new vector store...")
        documents = load_documents(URL)
        chunks = split_documents(documents)

        vector_store.add_documents(chunks)
        vector_store.persist()
        print(f"Stored {len(chunks)} chunks.")

    else:
        print("Loaded existing vector store.")

    return vector_store

def run_rag_query(query: str) -> str:
    db = build_or_load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)

    formatted_docs = "\n\n".join(
        f"Document {i+1}:\n{doc.page_content[:1000]}"
        for i, doc in enumerate(relevant_docs)
    )

    prompt = f"""
        Use ONLY the information from the documents below to answer the question.

        Question:
        {query}

        Documents:
        {formatted_docs}

        If the answer is not contained in the documents, say:
        "I don't have enough information to answer that question based on the provided documents."
    """

    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        max_retries=2,
    )

    messages = [
        SystemMessage(content="You are a precise RAG assistant."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    answer = run_rag_query("What is a RAG chain?")
    print("\nAnswer:\n")
    print(answer)
