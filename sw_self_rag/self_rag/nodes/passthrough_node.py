from typing import Dict

from self_rag.state import GraphState


def passthrough(state: GraphState) -> Dict:
    question = state["question"]
    documents = state["documents"]
    answer = state["generation"]

    return {"documents": documents, "question": question, "generation": answer}
