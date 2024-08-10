from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableSerializable


def history_aware_chain() -> RunnableSerializable:
    system = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    Output the question in a json format with the key 'question' and the reformulated question as value.
    """
    # history_aware_prompt = hub.pull("joeywhelan/rephrase")
    llm = ChatOllama(model="mistral", temperature=0.0)

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # return history_aware_prompt | llm | StrOutputParser()
    return history_aware_prompt | llm | JsonOutputParser()

