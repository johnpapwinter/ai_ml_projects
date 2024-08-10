from langgraph.graph import StateGraph, END

from self_rag.nodes import *
from self_rag.state import GraphState


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("websearch", web_search)
workflow.add_node("passthrough", passthrough)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate"
    }
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_hallucination,
    {
        "hallucination": "generate",
        "no_hallucination": "passthrough"
    }
)
workflow.add_conditional_edges(
    "passthrough",
    grade_answer,
    {
        "relevant": END,
        "no_relevant": "websearch"
    }
)


app_graph = workflow.compile()
