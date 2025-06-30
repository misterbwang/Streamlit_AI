import streamlit as st
import pandas as pd
import sqlalchemy
import requests

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
from tempfile import NamedTemporaryFile


st.set_page_config(page_title="🧠 RAG Chat with Ollama", layout="wide")
st.title("📁 Upload Files and Chat with RAG + Ollama")

st.sidebar.title("📤 Upload Data")
input_method = st.sidebar.radio("Choose input method:", ["Upload Files", "Connect to Database"])

temp_llm = "cognitivecomputations/dolphin-llama3.1"

dataframes = []
documents_to_index = []

if input_method == "Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload .csv, .xlsx, .txt, or .pdf",
        accept_multiple_files=True,
        type=["csv", "xlsx", "txt", "pdf"]
    )

    if uploaded_files:
        for file in uploaded_files:
            filename = file.name
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
                dataframes.append((filename, df))
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(file)
                dataframes.append((filename, df))
            elif filename.endswith(".txt"):
                with NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                loader = TextLoader(tmp_path)
                documents_to_index.extend(loader.load())
            elif filename.endswith(".pdf"):
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents_to_index.extend(loader.load())
            else:
                st.warning(f"Unsupported file format: {filename}")

    for name, df in dataframes:
        st.subheader(f"Preview: {name}")
        st.dataframe(df.head())

# --- Initialize embeddings and RAG components once ---
@st.cache_resource
def initialize_rag(docs, value = temp_llm):
    if not docs:
        # fallback dummy document
        from langchain.schema.document import Document
        docs = [Document(page_content="This is a placeholder. Please upload text or PDF files.")]
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="EntropyYue/jina-embeddings-v2-base-zh:latest")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model= value)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

rag_chain = initialize_rag(documents_to_index)

st.title("🧠 Chat with LLMs via Ollama")

# --- Sidebar: Model and Prompt Configuration ---
st.sidebar.title("🔧 Model & Prompt Settings")

ollama_model = st.sidebar.text_input("Ollama model name", value = temp_llm)

prompt_options = {
    "Helpful Assistant": "You are a helpful and friendly AI assistant.",
    "Technical Expert": "You are an expert in programming and data science.",
    "Creative Writer": "You are a creative and imaginative writer.",
    "Custom": "Custom"
}

selected_prompt = st.sidebar.selectbox("System Prompt", options=list(prompt_options.keys()))

if selected_prompt == "Custom":
    custom_prompt = st.sidebar.text_area("Enter your custom system prompt")
    system_prompt = custom_prompt.strip() or "You are a helpful assistant."
else:
    system_prompt = prompt_options[selected_prompt]

# --- Initialize Chat History with System Prompt ---
if "chat_history" not in st.session_state or st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hi! I'm Dolphin-LLaMA 3.1. How can I help you today?"}
    ]

# --- Chat UI and Input ---
chat_container = st.container()
user_input = st.chat_input("Type your message and press Enter...")

with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

# --- Handle Input ---
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Prepare payload
    payload = {
        "model": ollama_model,
        "messages": st.session_state.chat_history,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        assistant_msg = response.json()["message"]["content"]
    except Exception as e:
        assistant_msg = f"❌ Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").write(assistant_msg)