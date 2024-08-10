from self_rag.state import GraphState
from self_rag.chains.hallucination_chain import grade_hallucination_chain


def grade_hallucination(state: GraphState) -> str:
    print("Evaluating hallucination")
    hallucination_chain = grade_hallucination_chain()
    answer = state["generation"]
    documents = state["documents"]

    score = hallucination_chain.invoke({"documents": documents, "generation": answer})
    print(score)
    if score.binary_score.lower() == 'yes':
        print("Model hallucinating")
        return "hallucination"
    else:
        print("Found no hallucination")
        return "no_hallucination"

