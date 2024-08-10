from self_rag.state import GraphState
from self_rag.chains.grade_answer_chain import grade_answer_chain


def grade_answer(state: GraphState) -> str:
    print("Evaluating answer...")
    answer_relevant_chain = grade_answer_chain()
    question = state["question"]
    answer = state["generation"]

    score = answer_relevant_chain.invoke({"question": question, "generation": answer})
    if score.binary_score.lower() == 'yes':
        print("Answer relevant")
        return "relevant"
    else:
        print("Answer not relevant")
        return "not_relevant"



