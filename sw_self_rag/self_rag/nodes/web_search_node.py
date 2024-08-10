import os

from dotenv import load_dotenv
from typing import Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from self_rag.state import GraphState

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_TOKEN")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


def web_search(state: GraphState) -> Dict:
    print("Performing web Search...")
    web_search_tool = TavilySearchResults(k=5)

    question = state["question"]
    documents = state["documents"]
    doc_results = web_search_tool.invoke({"query": question})
    web_search_results = "\n".join([doc["content"] for doc in doc_results])
    web_search_results = Document(page_content=web_search_results)
    if documents is not None:
        documents.append(web_search_results)
    else:
        documents = [web_search_results]

    return {"documents": documents, "question": question}



