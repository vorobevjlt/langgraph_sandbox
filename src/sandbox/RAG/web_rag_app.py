from typing import List, Literal
from typing_extensions import TypedDict
import faiss
import bs4
from langchain_community.document_loaders import WebBaseLoader
import random
from langchain_classic import hub
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing_extensions import Annotated, List, TypedDict

from settings import settings

# with open('/Users/justlikethat/langgraph-course/resourses/artical_web_rag.txt','r') as file:
#     text = " ".join(line.rstrip() for line in file)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

all_splits = text_splitter.split_documents(docs)

total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

_ = vector_store.add_documents(all_splits)


class Search(TypedDict):
    """Search query only in english."""

    query: Annotated[str, ..., "Search query to run. Use english language"]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Query section to run the search in.",
    ]

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict, total=False):
    question: str
    query: Search
    context: List[Document]
    answer: str
    generate_count: int

class InputState(TypedDict):
    question: str

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}



def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter={"section": query["section"]},
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def cond_edge(state: State):
    return "analyze" if random.random() < 0.5 else END

graph_builder: StateGraph = StateGraph(State, input_schema=InputState).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

result = graph.invoke({"question": "Расскажи о Task Decomposition"})
print(result["answer"])