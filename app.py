import time
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# Load environment variables securely
load_dotenv()

# --- UI Styling & Config ---
st.set_page_config(page_title="Apple Brain", page_icon="🍎", layout="centered")

# Custom CSS for Premium, Energetic, Minimal Apple-like UI
st.markdown("""
<style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    /* App Background */
    .stApp {
        background-color: #0A0A0A;
        color: #F5F5F7;
    }

    /* Energetic Gradient Title */
    h1 {
        background: linear-gradient(120deg, #FF2A5F, #FF8A00, #A020F0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1.5px;
        font-size: 3.2rem !important;
        padding-bottom: 5px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #222 !important;
    }
    
    /* Premium File Uploader and Text Input */
    .stTextInput input, .stFileUploader {
        background-color: #1A1A1A !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
        color: white !important;
        transition: border 0.3s;
    }
    .stTextInput input:focus {
        border: 1px solid #FF8A00 !important;
    }
    
    /* Deep Rounded Chat Input Box */
    [data-testid="stChatInput"] {
        border-radius: 40px !important;
        border: 1px solid #333 !important;
        background-color: #111 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6) !important;
        overflow: hidden;
    }
    
    /* Energetic Buttons */
    .stButton>button {
        border-radius: 30px;
        border: none;
        background: linear-gradient(90deg, #FF2A5F, #A020F0);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(160, 32, 240, 0.4);
        color: white;
    }

    /* Pop Animation for Chat Bubbles */
    @keyframes popIn {
        0% { transform: scale(0.9) translateY(20px); opacity: 0; }
        70% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(1) translateY(0); opacity: 1; }
    }
    
    .stChatMessage {
        animation: popIn 0.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        background-color: #151515;
        border: 1px solid #222;
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    hr {
        border-color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍎 Apple Brain")
st.markdown("**Your intelligent, minimalist workspace.** Feed it knowledge, ask it anything.")

# --- Backend Initialization ---
@st.cache_resource
def init_brain():
    # Stable cloud embedding
    embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db")
    
    # Thinking disabled for instant streaming
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
    return vector_db, llm

vector_db, llm = init_brain()

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <h2 style="margin-bottom: 0px;">⚡ Neural Link</h2>
            <p style="color: #888; font-size: 0.9em;">Sync your external data</p>
        </div>
    """, unsafe_allow_html=True)

    
    # PDF Upload
    st.markdown("<p style='margin-bottom: 5px;'>📄 <b>Upload Document</b></p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="pdf_uploader")
    if uploaded_file is not None:
        if st.button("Sync PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            chunks = splitter.split_documents(documents)
            
            # Visual Progress Bar for PDF
            progress_bar = st.progress(0, text="Initiating sync...")
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = uploaded_file.name
                vector_db.add_documents([chunk])
                time.sleep(2) # Throttle to bypass Free Tier rate limits
                progress_bar.progress((i + 1) / len(chunks), text=f"Syncing chunk {i+1} of {len(chunks)}")
            
            os.remove(tmp_path) 
            st.success("✨ PDF successfully synced!")

# --- Interactive Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask your Apple Brain...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            results = vector_db.similarity_search(user_query, k=3)
            context = "\n".join([doc.page_content for doc in results])
            
            prompt = f"""
            You are a helpful AI assistant acting as a second brain. Use ONLY the following context to answer. 
            If the answer is not in the context, politely state that you do not have that information.
            
            Context: {context}
            Question: {user_query}
            Answer:
            """
            
            def stream_clean_text():
                for chunk in llm.stream(prompt):
                    if isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict) and "text" in item:
                                yield item["text"]
                    else:
                        yield chunk.content

            final_answer = st.write_stream(stream_clean_text())
            
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in results]))
            if sources:
                st.caption(f"Sources utilized: {', '.join(sources)}")
            
    st.session_state.messages.append({"role": "assistant", "content": final_answer})