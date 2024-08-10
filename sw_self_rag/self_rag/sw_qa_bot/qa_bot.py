from self_rag.chains.history_aware_chain import history_aware_chain
from self_rag.graph import app_graph


def get_qa_bot():
    history_chain = history_aware_chain()
    self_rag_chain = app_graph

    return history_chain | self_rag_chain

