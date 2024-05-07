from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from prompts import *


class GMHelper:
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings()
        self._chroma_directory = "doc_chroma/"
        self._vector_db = Chroma(
            collection_name="races_monsters",
            persist_directory= self._chroma_directory,
            embedding_function=self._embeddings,
        )
        self._llm = ChatOllama(
            model="llama3",
            temperature=0,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        self._retriever = self._vector_db.as_retriever()

    def _get_context_retriever_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUAL_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        return create_history_aware_retriever(self._llm, self._retriever, prompt)

    def _get_conversational_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(llm=self._llm, prompt=prompt)
        return create_retrieval_chain(self._get_context_retriever_chain(), qa_chain)

    def get_response(self, user_input: str, chat_history: list):
        conversation_rag_chain = self._get_conversational_rag_chain()

        return conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
        })['answer']


