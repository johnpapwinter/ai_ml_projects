from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable


def rag_chain() -> RunnableSerializable:
    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model="mistral", temperature=0.0)

    return rag_prompt | llm | StrOutputParser()

