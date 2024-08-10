from typing import Dict

from self_rag.state import GraphState
from self_rag.chains.rag_chain import rag_chain


def generate_answer(state: GraphState) -> Dict:
    print("Generating answer...")
    question = state["question"]
    documents = state["documents"]

    answer = rag_chain().invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": answer}
