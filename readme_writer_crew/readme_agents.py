import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai_tools.tools import GithubSearchTool
from crewai_tools import tool
from crewai import Agent

load_dotenv()


@tool("Markdown Exporter")
def export_to_markdown(text: str) -> str:
    """This is a tool that exports a string to a Markdown file."""
    with open("README.md", "w") as file:
        file.write(text)
    return "OK"


class DocumentationAgents:
    def __init__(self, repository_url):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.llm = ChatGroq(temperature=0,
                            groq_api_key=self.groq_api_key,
                            model_name="llama3-70b-8192")
        self.github_tool = GithubSearchTool(
            gh_token=self.github_token,
            github_repo=repository_url,
            content_types=["code"],
            config=dict(
                llm=dict(
                    provider="ollama",
                    config=dict(
                        model="mistral",
                        temperature=0.0,
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="all-MiniLM-L6-v2",
                    )
                )
            )
        )

    def technical_document_writer(self) -> Agent:
        return Agent(
            role="Technical Document Writer",
            goal="Read a codebase and craft a document which precisely describes how to use it.",
            backstory='''Your expertise lies in reading all the files in a GitHub repository.
            You have a knack for dissenting the complex structures of a software project and writing a simple
            technical document that describes what that project does and how can users use it.
            ''',
            allow_delegation=False,
            tools=[self.github_tool],
            llm=self.llm,
            verbose=True
        )

    def markdown_writer(self):
        return Agent(
            role="Markdown Writer",
            goal="Read technical documentation and create a simple and concise README.md file from it.",
            backstory='''You are know for being able to summarize a complex technical document describing
            what a software project does and creating a short and concise README.md file from it.
            ''',
            allow_delegation=False,
            tools=[export_to_markdown],
            llm=self.llm,
            verbose=True
        )

