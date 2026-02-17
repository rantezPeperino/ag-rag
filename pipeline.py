from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


class RAGGraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


def build_app(llm, retriever, prompt):
    def retrieve_documents_node(state: RAGGraphState) -> RAGGraphState:
        question = state["question"]
        documents = retriever.invoke(question)
        return {"question": question, "documents": documents, "generation": ""}

    def generate_response_node(state: RAGGraphState) -> RAGGraphState:
        question = state["question"]
        documents = state["documents"]

        context = "\n\n".join([doc.page_content for doc in documents])
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": context, "question": question})

        return {"question": question, "documents": documents, "generation": generation}

    workflow = StateGraph(RAGGraphState)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("generate", generate_response_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_query(app, question: str):
    inputs = {"question": question}
    last_state = None
    for state in app.stream(inputs):
        last_state = state
    return last_state
