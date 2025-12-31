import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain / Groq imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# Load local .env (only used when developing locally)
load_dotenv()

st.set_page_config(page_title="Resume Skill Search", page_icon="📑")
st.title("📑 Resume Skill Search Bot (Groq)")

# === Secure API key loading (Streamlit Secrets preferred) ===
# Streamlit Cloud: add key under Settings → Secrets as GROQ_API_KEY
groq_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = os.getenv("GROQ_API_KEY")

# Debug: show presence (do NOT print the actual key)
st.write("Groq key found:", bool(groq_key))
if groq_key:
    try:
        masked = "****" + groq_key[-4:]
    except Exception:
        masked = "****"
    st.caption(f"Groq key (masked): {masked}")

if not groq_key:
    st.error(
        "GROQ_API_KEY not found. Add it to Streamlit Secrets (preferred) or create a local .env file with GROQ_API_KEY."
    )
    st.stop()

# === Setup LLM and embeddings (wrapped in try/except for auth errors) ===
try:
    llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant", temperature=0.3)
except Exception as e:
    st.error("Failed to initialize Groq LLM. Check your API key and network.")
    st.exception(e)
    st.stop()

# -----------------------------------------------------------------------------
# UI: upload resumes and index them (Chroma)
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

# NOTE: FakeEmbeddings are placeholders — replace with real embeddings for production
embeddings = FakeEmbeddings(size=384)
PERSIST_DIR = "chroma_db"

if uploaded_files:
    documents = []
    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getbuffer())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    st.success("✅ Resumes indexed successfully!")

# Search UI
if os.path.exists(PERSIST_DIR):
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    query = st.text_input("Search skills (eg: Python, Java, ML)")
    if query:
        docs = vectordb.similarity_search(query, k=5)
        st.subheader("📌 Matching Candidates")
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        From the following resume content, identify candidates
        matching these skills: {query}

        Resume Content:
        {context}
        """

        # Call the model safely
        try:
            # ChatGroq provides `.invoke(...)` that returns an object with .content
            # or you can use llm.create / llm.call depending on library version.
            result = llm.invoke(prompt)
            # result may be a Response-like object; show text safely.
            text = getattr(result, "content", str(result))
            st.success(text)
        except Exception as e:
            # If this is an authentication error it will be caught here:
            st.error("Error while calling Groq API. Check your GROQ_API_KEY and account limits.")
            st.exception(e)
