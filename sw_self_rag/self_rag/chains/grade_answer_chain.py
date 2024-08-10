from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSerializable


class QuestionAnswered(BaseModel):
    """Binary score for checking whether the answer resolves the question"""
    binary_score: str = Field(
        description="Does the answer resolve the question, 'Yes' or 'No'"
    )


def grade_answer_chain() -> RunnableSerializable:
    llm = ChatOllama(model="mistral", temperature=0.0)

    system = """You are a grader assessing whether an answer addresses / resolves a question. \n 
         Give a binary score 'Yes' or 'No'. 'Yes' means that the answer resolves the question.
         Answer in json format with a key 'binary_score' and the value of the binary score."""
    grade_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
        ]
    )

    return grade_answer_prompt | llm | PydanticOutputParser(pydantic_object=QuestionAnswered)
