import streamlit as st
import pandas as pd
import sqlalchemy
import requests


st.sidebar.title("Data Input")

# Select input method
input_method = st.sidebar.radio(
    "Choose your data source:",
    ("Upload Files", "Connect to Database")
)

dataframes = []

if input_method == "Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more files",
        accept_multiple_files=True,
        type=["csv", "xlsx"]
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file format: {file.name}")
                continue

            dataframes.append((file.name, df))

        for name, df in dataframes:
            st.subheader(f"Preview: {name}")
            st.dataframe(df.head())

elif input_method == "Connect to Database":
    st.sidebar.subheader("Database Credentials")

    db_type = st.sidebar.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"])
    host = st.sidebar.text_input("Host", "localhost")
    port = st.sidebar.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306")
    database = st.sidebar.text_input("Database Name")
    user = st.sidebar.text_input("User")
    password = st.sidebar.text_input("Password", type="password")

    connect_button = st.sidebar.button("Connect")

    if connect_button:
        try:
            if db_type == "PostgreSQL":
                uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            elif db_type == "MySQL":
                uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            elif db_type == "SQLite":
                uri = f"sqlite:///{database}"  # For SQLite, `database` is the path to the file

            engine = sqlalchemy.create_engine(uri)
            tables = engine.table_names()
            selected_table = st.sidebar.selectbox("Select a Table", tables)

            if selected_table:
                df = pd.read_sql_table(selected_table, con=engine)
                st.subheader(f"Preview: {selected_table}")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to connect or load data: {e}")


st.title("🧠 Chat with LLMs via Ollama")

# --- Sidebar: Model and Prompt Configuration ---
st.sidebar.title("🔧 Model & Prompt Settings")

ollama_model = st.sidebar.text_input("Ollama model name", value="cognitivecomputations/dolphin-llama3.1")

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