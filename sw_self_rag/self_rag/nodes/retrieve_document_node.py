from typing import Dict

from self_rag.state import GraphState
from self_rag.ingestion import get_vectorstore_retriever


def retrieve_documents(state: GraphState) -> Dict:
    print("Retrieving documents...")
    retriever = get_vectorstore_retriever()

    question = state["question"]
    documents = retriever.invoke(question)
    print("*" * 30)
    print(documents)
    print("*" * 30)

    return {"documents": documents, "question": question}


