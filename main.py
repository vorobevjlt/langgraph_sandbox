from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ---------- 1) ingest one site ----------
loader = RecursiveUrlLoader(
    "https://avito.ru",
    max_depth=2,
    prevent_outside=True,
)

pages = loader.load()
print(pages)
# # Optional: enrich metadata before splitting
# for d in pages:
#     d.metadata["site"] = "example-docs"
#     d.metadata["section"] = d.metadata.get("title", "")

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=200,
# )

# chunks = splitter.split_documents(pages)

# # give every chunk a stable citation id
# for i, d in enumerate(chunks, start=1):
#     d.metadata["chunk_id"] = i

# # ---------- 2) build retriever ----------
# vectorstore = InMemoryVectorStore.from_documents(
#     documents=chunks,
#     embedding=OpenAIEmbeddings(),
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


# # ---------- 3) LangGraph state ----------
# class SiteQAState(TypedDict):
#     question: str
#     docs: list
#     answer: str
#     citations: list


# def retrieve_node(state: SiteQAState):
#     docs = retriever.invoke(state["question"])
#     return {"docs": docs}


# def answer_node(state: SiteQAState):
#     context_blocks = []
#     citations = []

#     for idx, d in enumerate(state["docs"], start=1):
#         url = d.metadata.get("source", "")
#         title = d.metadata.get("title", "")
#         chunk_id = d.metadata.get("chunk_id")

#         context_blocks.append(
#             f"[{idx}] TITLE: {title}\n"
#             f"URL: {url}\n"
#             f"CHUNK_ID: {chunk_id}\n"
#             f"CONTENT:\n{d.page_content}"
#         )

#         citations.append(
#             {
#                 "ref": idx,
#                 "url": url,
#                 "title": title,
#                 "chunk_id": chunk_id,
#             }
#         )

#     prompt = f"""
# You answer questions using ONLY the provided context.

# Rules:
# - Every important claim must cite one or more refs like [1] or [2][4].
# - Never cite anything not in the context.
# - If the context is insufficient, say you don't know.
# - Prefer multiple refs when the answer is supported by more than one page.

# Question:
# {state["question"]}

# Context:
# {chr(10).join(context_blocks)}
# """.strip()

#     llm = ChatOpenAI(model="gpt-5")
#     answer = llm.invoke(prompt).content

#     return {
#         "answer": answer,
#         "citations": citations,
#     }


# # ---------- 4) graph ----------
# builder = StateGraph(SiteQAState)
# builder.add_node("retrieve", retrieve_node)
# builder.add_node("answer", answer_node)

# builder.add_edge(START, "retrieve")
# builder.add_edge("retrieve", "answer")
# builder.add_edge("answer", END)

# app = builder.compile()

# # ---------- 5) run ----------
# result = app.invoke({"question": "How does authentication work?"})

# print(result["answer"])
# print(result["citations"])