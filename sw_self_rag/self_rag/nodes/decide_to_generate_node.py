from self_rag.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    print("Deciding whether to generate...")
    web_search = state["web_search"]

    if web_search == "Yes":
        print("Web search required")
        return "websearch"
    else:
        print("Proceeding to generation")
        return "generate"



