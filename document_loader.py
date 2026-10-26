import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_document(file_path: str):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_spiltter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    splits = text_spiltter.split_documents(docs)
    return splits