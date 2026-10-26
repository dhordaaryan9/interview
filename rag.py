from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from vector_store import get_retriver
from config import QA_SYSTEM_PROMPT

def get_llm():

    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

def build_rag_chain():

    llm = get_llm()
    retriever = get_retriver()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(question_answer_chain)
    
    return rag_chain

def get_answer(query: str, chat_history: list):

    rag_chain = build_rag_chain()
    
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response["answer"], response["context"]
