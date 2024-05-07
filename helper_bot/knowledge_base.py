from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class KnowledgeBase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.chroma_directory = "doc_chroma/"
        self.collection_name = "races_monsters"
        self.chunk_size = 1500
        self.chunk_overlap = 150

    def upload_document(self, documents):
        # pass through data pipeline
        vector_db = Chroma.from_documents(
            collection_name=self.collection_name,
            documents=self.split_documents(documents=documents),
            embeddings=self.embeddings,
            persist_directory=self.chroma_directory,
        )

        vector_db.persist()

    def update_document(self, document):
        # pass through data pipeline
        pass

    def delete_document(self, document):
        pass

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_documents(documents=documents)
