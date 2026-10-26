import streamlit as st
import os

from document_loader import load_and_split_document 
from vector_store import create_vector_store 
from rag import get_answer 


st.title("ChatBot")

# 1. File Upload Section
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily so the loader can read it
    file_path = "temp_uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # 2. Process Document Button
    if st.button("Process PDF"):
        with st.spinner("Splitting text and saving to vector store..."):
            # Load and split using document_loader.py[cite: 1]
            splits = load_and_split_document(file_path) 
            
            # Save to Chroma DB using vector_store.py
            create_vector_store(splits) 
            
            st.success("PDF processed successfully! You can now ask questions.")

# 3. Chat / Query Section
st.divider()
st.subheader("Ask Questions")

query = st.text_input("Enter your question here:")

if st.button("Submit Question"):
    if query:
        with st.spinner("Generating answer..."):
            # Generate answer using rag.py (passing empty chat history for simplicity)
            answer, context = get_answer(query=query, chat_history=[])
            
            st.write("### Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question first.")