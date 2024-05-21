from io import BytesIO
from typing import Any, List

from langchain_core.document_loaders import Blob
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser


class BytesIOPyMuPDFLoader(PyMuPDFLoader):
    def __init__(self, pdf_stream: BytesIO, *, extract_images: bool = False, **kwargs: Any) -> None:
        try:
            import fitz
        except ImportError:
            raise ImportError("Please install PyMuPDF with pip install pymupdf")
        self.pdf_stream = pdf_stream
        self.extract_images = extract_images
        self.text_kwargs = kwargs

    def load(self, **kwargs: Any) -> List[Document]:
        if kwargs:
            print(f"Received {kwargs}")
        text_kwargs = {**self.text_kwargs, **kwargs}
        blob = Blob.from_data(self.pdf_stream.getvalue(), path="stream")

        parser = PyMuPDFParser(text_kwargs=text_kwargs, extract_images=self.extract_images)

        return parser.parse(blob=blob)


