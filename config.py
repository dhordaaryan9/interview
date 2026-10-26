import os

CHUNK_SIZE=1000
CHUNK_OVERLAP=200

VECTOR_DB_DIR = "./chromadb"
UPLOADS_DIR = "./uploads"


QA_SYSTEM_PROMPT = """You are a helpful HR assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say "I cannot find that information in the provided HR policy documents".
Always cite the exact source and page numbers from the provided context in your answer. Do not guess or make up facts."""