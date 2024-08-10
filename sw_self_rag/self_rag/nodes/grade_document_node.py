from typing import Dict

from self_rag.state import GraphState
from self_rag.chains.grade_documents_chain import grade_documents_chain


def grade_documents(state: GraphState) -> Dict:
    print("Grading documents...")
    retrieval_grader = grade_documents_chain()
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = "No"

    for document in documents:
        score = retrieval_grader.invoke({"question": question, "document": document})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("Document relevant")
            filtered_documents.append(document)
        else:
            print("Document not relevant, will perform web search")
            web_search = "Yes"
            continue

    return {
        "documents": filtered_documents,
        "question": question,
        "web_search": web_search,
    }


