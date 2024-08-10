from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSerializable


class Hallucination(BaseModel):
    """Binary score for checking whether the answer is grounded in / supported by a set of facts"""
    binary_score: str = Field(
        description="Is the answer grounded in / supported by the set of facts, 'Yes' or 'No'"
    )


def grade_hallucination_chain() -> RunnableSerializable:
    llm = ChatOllama(model="mistral", temperature=0.0)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'Yes' or 'No'. 'Yes' means that the answer is grounded in / supported by the set of facts.
     Answer in json format with a key 'binary_score' and the value of the binary score."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
        ]
    )

    return hallucination_prompt | llm | PydanticOutputParser(pydantic_object=Hallucination)

