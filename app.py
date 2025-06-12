import streamlit as st
import uuid
from st_copy_to_clipboard import st_copy_to_clipboard
import torch
torch.classes.__path__ = []

# Import the updated utility functions
from utils import (
    load_embedding_model,
    load_moondream_model,
    get_qdrant_client,
    process_and_embed_document,
    search_qdrant,
    create_passage_from_snippets
)

# --- App Configuration ---
st.set_page_config(
    page_title="Document RAG for Question Context Search",
    page_icon="✨",
    layout="wide"
)

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# --- Main App UI ---
st.title("✨ Document RAG for Question Context Search")
st.markdown(f"**Session ID:** `{st.session_state.session_id}` (Your uploads are isolated to this session)")

# --- Load Models and Clients ---
try:
    embed_model = load_embedding_model()
    moondream_model = load_moondream_model()
    qdrant_client = get_qdrant_client()
except ValueError as e:
    st.error(f"Configuration Error: {e}. Please check your .env file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred on startup: {e}")
    st.stop()


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDFs only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.header("2. Search Parameters")
    marks_for_question = st.number_input("Marks for the question:", min_value=1, max_value=50, value=5)
    default_k = marks_for_question * 2
    k_snippets = st.number_input("Number of snippets to retrieve (k):", min_value=1, max_value=40, value=min(default_k, 40))
    similarity_threshold = st.slider("Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.header("3. Enter Your Question")
    user_query = st.text_area("What would you like to ask about your documents?", height=150)

    search_button = st.button("Search", type="primary")

# --- Document Processing Logic ---
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            if process_and_embed_document(file, embed_model, moondream_model, qdrant_client, st.session_state.session_id):
                st.session_state.processed_files.append(file.name)
        else:
            st.info(f"'{file.name}' has already been processed in this session.")

# --- Search and Display Logic ---
if search_button:
    if not st.session_state.processed_files:
        st.warning("Please upload and process at least one document before searching.")
    elif not user_query:
        st.warning("Please enter a question to search.")
    else:
        with st.spinner("Searching for relevant snippets..."):
            st.session_state.search_results = search_qdrant(
                user_query, embed_model, qdrant_client, st.session_state.session_id, k_snippets, similarity_threshold
            )

# --- Display Results ---
if st.session_state.search_results is not None:
    passage = create_passage_from_snippets(st.session_state.search_results)
    
    st.subheader("📝 Combined Passage from Retrieved Snippets")
    
    if not st.session_state.search_results:
        st.warning("No relevant snippets were found based on your query and thresholds. Try adjusting the similarity threshold or your question.")
    else:
        st.text_area(
            "The following passage is constructed from the most relevant snippets found in your documents:",
            passage, height=400, key="passage_display"
        )
        st_copy_to_clipboard(passage, "Copy Passage to Clipboard")
        
        with st.expander("View Individual Snippets and Scores (for verification)"):
            for result in st.session_state.search_results:
                st.markdown(f"""---
                **Source:** `{result.payload.get('source_file', 'N/A')}` | **Score:** `{result.score:.4f}`
                
                > {result.payload.get('text', 'No text available')}""")