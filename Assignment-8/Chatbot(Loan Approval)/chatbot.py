import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI

os.makedirs("data", exist_ok=True)

def load_data_files():
    """Load various document types including dataset CSV"""
    documents = []
    
    # Load PDF files
    pdf_path = 'data/pdfs'
    if os.path.exists(pdf_path) and os.listdir(pdf_path):
        loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(loader.load())
    
    # Load text files
    txt_path = 'data/texts'
    if os.path.exists(txt_path) and os.listdir(txt_path):
        loader = DirectoryLoader(txt_path, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(loader.load())
    
    # Load CSV dataset (loan approval data)
    csv_path = 'data/loan_approval_data.csv'
    if os.path.exists(csv_path):
        loader = CSVLoader(file_path=csv_path)
        documents.extend(loader.load())
    
    if not documents:
        st.warning("No documents found in data directory. Please upload files.")
        return None
    
    return documents

def process_documents(documents):
    """Process and split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Process each document individually to maintain metadata
    processed_chunks = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        processed_chunks.extend(chunks)
    return processed_chunks

def setup_vector_store(chunks, embeddings_model='hkunlp/instructor-xl'):
    """Create embeddings and vector store"""
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store for future use
    vector_store_path = "data/vector_store"
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    
    return vector_store

def initialize_chatbot(vector_store, openai_api_key):
    """Initialize the chatbot components"""
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )

def save_uploaded_files(uploaded_files):
    """Save uploaded files to appropriate directories"""
    for file in uploaded_files:
        file_extension = file.name.split('.')[-1].lower()
        file_type_dir = ""
        
        if file_extension == "pdf":
            file_type_dir = "pdfs"
        elif file_extension == "txt":
            file_type_dir = "texts"
        elif file_extension == "csv":
            file_type_dir = ""  # Save CSV directly in data dir
        
        save_dir = os.path.join("data", file_type_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, file.name) if file_type_dir else os.path.join("data", file.name)
        
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
    
    st.sidebar.success(f"Saved {len(uploaded_files)} file(s)")

def main():
    st.set_page_config(page_title="Loan Approval AI Assistant", page_icon="💰")
    st.title("💰 Loan Approval AI Assistant")
    st.markdown("Ask questions about loan applications and approval criteria")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration ⚙️")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        hf_token = st.text_input("HuggingFace Hub Token", type="password")
        
        st.markdown("### Document Management 📂")
        uploaded_files = st.file_uploader(
            "Upload loan documents (PDF/TXT/CSV)",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            save_uploaded_files(uploaded_files)
        
        if st.button("⏳ Process Documents"):
            if not openai_api_key:
                st.error("Please provide your OpenAI API key")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        documents = load_data_files()
                        if documents:
                            chunks = process_documents(documents)
                            st.session_state.vector_store = setup_vector_store(chunks)
                            st.session_state.qa_chain = initialize_chatbot(
                                st.session_state.vector_store,
                                openai_api_key
                            )
                            st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    # Main chat interface
    if st.session_state.vector_store:
        user_question = st.chat_input("Ask about loan approvals...")
        
        if user_question:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get response
            with st.spinner("Generating response..."):
                try:
                    result = st.session_state.qa_chain({"question": user_question})
                    answer = result['answer']
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Show source documents if available
                    if result.get('source_documents'):
                        with st.expander("📄 Reference Documents"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.markdown(f"**Document {i}**: `{doc.metadata.get('source', 'unknown')}`")
                                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                                if i < len(result['source_documents']):
                                    st.divider()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        # Display chat history
        for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    else:
        st.info("Please upload and process documents to begin chatting")

if __name__ == "__main__":
    main()
