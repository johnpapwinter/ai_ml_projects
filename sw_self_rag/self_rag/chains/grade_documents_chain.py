from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSerializable


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(
        description="Are the documents relevant to the question, 'Yes' or 'No'"
    )


def grade_documents_chain() -> RunnableSerializable:
    llm = ChatOllama(model="mistral", temperature=0.0)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'Yes' or 'No' score to indicate whether the document is relevant to the question.\n
        Answer in json format with a key 'binary_score' and the value of the binary score."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )

    return grade_prompt | llm | PydanticOutputParser(pydantic_object=GradeDocuments)
