from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


def create_vectorstore() -> None:
    urls = [
        "https://starwars.fandom.com/wiki/Star_Wars:_Episode_I_The_Phantom_Menace",
        "https://starwars.fandom.com/wiki/Star_Wars:_Episode_II_Attack_of_the_Clones",
        "https://starwars.fandom.com/wiki/Star_Wars:_Episode_III_Revenge_of_the_Sith",
    ]

    docs = [WebBaseLoader(urls).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    docs_split = text_splitter.split_documents(docs_list)

    Chroma.from_documents(
        documents=docs_split,
        collection_name='rag-chroma',
        embedding=HuggingFaceEmbeddings(),
        persist_directory='./chroma'
    )


def get_vectorstore_retriever() -> VectorStoreRetriever:
    return Chroma(
        collection_name='rag-chroma',
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory='./chroma'
    ).as_retriever()
