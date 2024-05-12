import uuid
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class KnowledgeBase:
    def __init__(self):
        self.data_path = 'data'
        self.collection_name = 'srd_collection'
        self.client = chromadb.PersistentClient(path=self.data_path)
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            get_or_create=True
        )

    def create_document(self, filename: str, document: list[Document]) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = splitter.split_documents(documents=document)
        metadata = {"filename": filename}
        self.collection.add(
            ids=[str(uuid.uuid1()) for _ in splits],
            documents=[chunk.page_content for chunk in splits],
            metadatas=[metadata for _ in splits]
        )

    def update_document(self, filename: str, new_document) -> None:
        self.collection.delete(where={"filename": filename})
        self.create_document(filename, new_document)

    def delete_document(self, filename: str) -> None:
        self.collection.delete(where={"filename": filename})

    def get_filenames(self):
        all_chunks = self.collection.get()
        chunk_metadata = [chunk for chunk in all_chunks['metadatas']]
        filenames = set([name['filename'] for name in chunk_metadata])
        # return [name['filename'] for name in chunk_metadata]
        return set(filenames)

    def _parse_document(self, document) -> list[Document]:
        return PyPDFLoader(document).load()

