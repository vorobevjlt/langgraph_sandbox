from typing import List
from typing_extensions import TypedDict
import faiss
import random
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from settings import settings

with open('/Users/justlikethat/langgraph-course/resourses/artical_web_rag.txt','r') as file:
    text = " ".join(line.rstrip() for line in file)

llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY,
        base_url="https://api.proxyapi.ru/openai/v1",
        max_retries=2,
    )


embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                              api_key=settings.OPENAI_API_KEY,
                              base_url='https://api.proxyapi.ru/openai/v1')

dimention_embedding = len(embeddings.embed_query("test"))
index = faiss.IndexFlatL2(dimention_embedding)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

docs = [Document(page_content=text)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    generate_count: int


class InputState(TypedDict):
    question: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def analyze_query(state: State):
    last_answer = state[answer][-1]

    followup_prompt = f"""
    Based on this answer:
    {last_answer}
    Generate one follow up question thit might be logically arise
    """
    new_question = llm.invoke(followup_prompt).content
    return {"question": new_question}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    prompt = f"""
    Use the following context to answer the question. If you not find any relevant answer say: "Sorry i cant find relevant information".

    Context:
    {docs_content}

    Question:
    {state['question']}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "generate_count": state.get("generate_count", 0 ) + 1
        }

def cond_edge(state: State):
    if random.random() < 0.5:
        return "generate"
    else:
        return END

graph_builder = StateGraph(State, input_schema=InputState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("analyze", analyze_query)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("analyze", "retrieve")
graph_builder.add_conditional_edges(
    "generate",
    cond_edge,
    {
        "generate": "generate",
        END: END
    }
)

graph = graph_builder.compile()

result = graph.invoke({"question": "What is self-attention?"})
print(result["answer"])
print(result["generate_count"])
