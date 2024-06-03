from crewai import Task, Crew, Process
from readme_agents import DocumentationAgents


github_repository = "PLACEHOLDER"
agents = DocumentationAgents(github_repository)

read_repository = Task(
    description='''Read the files in the repository and create a document that details
    the purpose and function of the project for a user as well as how to use that project
    (APIs, user interfaces if any etc.)  
    ''',
    expected_output='''A technical document that describes the purpose of a project
    and how to use it.''',
    agent=agents.technical_document_writer()
)

export_markdown = Task(
    description='''Using a technical document that describes the purpose of a project,
    its functions and how to use it (APIs, user interfaces etc.), create a short
    and concise markdown document that will serve as a GitHub README.md file
    ''',
    expected_output='''A short markdown file that describes the purpose of the project''',
    agent=agents.markdown_writer()
)


crew = Crew(
    agents=[agents.technical_document_writer(), agents.markdown_writer()],
    tasks=[read_repository, export_markdown],
    verbose=2,
    process=Process.sequential
)


result = crew.kickoff()
print("-" * 30)
print(result)
