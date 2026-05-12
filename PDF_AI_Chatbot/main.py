import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------------------------
# LOAD ENV VARIABLES
# ---------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS (CHAT UI DESIGN)
# ---------------------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.upload-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.chat-container {
    background-color: #111827;
    padding: 15px;
    border-radius: 15px;
}

.user-msg {
    background-color: #2563eb;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
}

.bot-msg {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
}

.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

.stFileUploader {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
}

.stButton button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 45px;
    border: none;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# TITLE
# ---------------------------------
st.markdown('<div class="title">🤖 RAG PDF Chatbot</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Upload a PDF and chat with your document</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# SESSION STATE
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    process_btn = st.button("Process PDF")

# ---------------------------------
# PROCESS PDF
# ---------------------------------
if process_btn and uploaded_pdf:

    with st.spinner("Processing PDF..."):

        # Read PDF
        reader = PdfReader(uploaded_pdf)

        text = ""

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

        # Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        # Embeddings
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY
        )

        # Vector DB
        vector_db = FAISS.from_texts(
            chunks,
            embeddings
        )

        st.session_state.vector_db = vector_db

    st.success("✅ PDF processed successfully!")

# ---------------------------------
# DISPLAY CHAT MESSAGES
# ---------------------------------
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="bot-msg">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ---------------------------------
# CHAT INPUT
# ---------------------------------
query = st.chat_input("Ask a question about your PDF...")

# ---------------------------------
# HANDLE USER QUERY
# ---------------------------------
if query:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    st.markdown(
        f'<div class="user-msg">🧑 {query}</div>',
        unsafe_allow_html=True
    )

    # Check PDF uploaded
    if st.session_state.vector_db is None:

        answer = "⚠️ Please upload and process a PDF first."

    else:

        with st.spinner("Thinking..."):

            # Similarity Search
            docs = st.session_state.vector_db.similarity_search(
                query,
                k=3
            )

            context = "\n\n".join(
                d.page_content for d in docs
            )

            context = context[:4000]

            # LLM
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model="gpt-3.5-turbo",
                temperature=0
            )

            prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
say:
"I could not find the answer in the PDF."

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

            answer = response.content

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Display assistant message
    st.markdown(
        f'<div class="bot-msg">🤖 {answer}</div>',
        unsafe_allow_html=True
    )