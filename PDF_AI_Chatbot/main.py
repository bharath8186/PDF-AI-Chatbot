import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


st.title("📄 RAG PDF Chatbot")


# -------------------------
# API KEY INPUT
# -------------------------
api_key = st.text_input("Enter OpenAI API Key", type="password")


# -------------------------
# PDF UPLOAD
# -------------------------
pdf = st.file_uploader("Upload PDF", type="pdf")


if pdf and api_key:

    # Read PDF
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()


    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)


    # Embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Vector DB
    db = FAISS.from_texts(chunks, embeddings)


    # LLM
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )


    query = st.text_input("Ask a question about the PDF:")

    if query:

        docs = db.similarity_search(query, k=2)

        context = "\n\n".join(d.page_content for d in docs)
        context = context[:3000]  # safety

        prompt = f"""
Answer using only the context below.

Context:
{context}

Question: {query}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)