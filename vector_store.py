from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from config import VECTOR_DB_DIR

def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def create_vector_store(documents):

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    return vectorstore

def get_vector_store():

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def get_retriver():

    vectorstore = get_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k":4})