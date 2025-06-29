import streamlit as st
import uuid
import io
import os
import zipfile
from datetime import datetime
from st_copy_to_clipboard import st_copy_to_clipboard
import torch

# Import the updated utility functions
from utils import (
    load_embedding_model,
    load_moondream_model,
    load_marker_model,
    get_qdrant_client,
    process_and_embed_document,
    prepare_document_for_download,
    search_qdrant,
    create_passage_from_snippets
)

# --- App Configuration ---
st.set_page_config(
    page_title="Document Processing Platform",
    page_icon="🥷",
    layout="wide"
)

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = [] # Now a list to hold multiple results


# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🛠️ Document Processing Platform")
    st.markdown("---")
    # This header makes the tool selection more explicit.
    st.header("Select a Tool")
    app_mode = st.radio(
        "Navigation", # The label for the radio button group
        ("RAG Search", "Document Extraction Pipeline"),
        label_visibility="collapsed", # Hide the "Navigation" label to keep UI clean
        help="""
        **RAG Search**: Ask questions about your PDFs.
        **Document Extraction Pipeline**: Convert PDFs to Markdown and download images.
        """
    )
    st.markdown("---")

# =============================================================================
# --- RAG SEARCH TOOL ---
# =============================================================================
if app_mode == "RAG Search":
    st.title("🧠 RAG for Question-Context Search")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}` (Your uploads are isolated to this session)")

    # --- Sidebar Inputs for RAG (part 1) ---
    with st.sidebar:
        st.header("1. Document Configuration")
        uploaded_files = st.file_uploader(
            "Upload PDFs to create a searchable knowledge base.",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_uploader"
        )
        page_range_input_rag = st.text_input(
            "Page Range (e.g., '0,5-10,20'):",
            help="Specify which pages to process (0-indexed). Leave empty for all pages. Changing this will require reprocessing."
        )

    # --- Load RAG-specific Models (with page range config) ---
    try:
        embed_model = load_embedding_model()
        moondream_model, moondream_limiter = load_moondream_model()
        # Pass page range to model loader. Caching handles re-initialization.
        marker_converter = load_marker_model(page_range_input_rag) 
        qdrant_client, collection = None, None 
        
    except ValueError as e:
        st.error(f"RAG Tool Configuration Error: {e}. Please check your .env file for all required API keys and URLs.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load models. Error: {e}")
        st.stop()

    # --- Sidebar Inputs for RAG (part 2) ---
    with st.sidebar:
        st.header("2. Search Parameters")
        marks_for_question = st.number_input("Marks for the question:", min_value=1, max_value=50, value=5)
        default_k = marks_for_question * 2
        k_snippets = st.number_input("Number of snippets to retrieve (k):", min_value=1, max_value=40, value=min(default_k, 40))
        similarity_threshold = st.slider("Similarity Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.header("3. Enter Your Question")
        user_query = st.text_area("What would you like to ask about your documents?", height=150, key="rag_query")
        search_button = st.button("Search", type="primary", key="rag_search_button")

    # --- RAG Processing Logic ---
    if uploaded_files:
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if files_to_process:
            st.toast(f"Found {len(files_to_process)} new document(s) to process.")
            
            # Lazily initialize Qdrant client and collection ONLY when a file is processed
            try:
                qdrant_client, collection = get_qdrant_client(st.session_state.session_id)
            except ValueError as e:
                st.error(f"Qdrant Client Initialization Error: {e}. Please ensure QDRANT_URL and QDRANT_API_KEY are set.")
                st.stop()

            for file in files_to_process:
                st.info(f"Processing '{file.name}'...")
                # The marker_converter is now pre-configured with the page range.
                if process_and_embed_document(file, embed_model, moondream_model, moondream_limiter, marker_converter, qdrant_client, collection): 
                    st.session_state.processed_files.append(file.name)
            st.success("All new files have been processed and indexed into the knowledge base.")
        
        st.markdown("##### Indexed Documents in this Session:")
        for file_name in st.session_state.processed_files:
            st.markdown(f"- `{file_name}`")


    if search_button:
        if not st.session_state.processed_files:
            st.warning("Please upload and process at least one document before searching.")
        elif not user_query:
            st.warning("Please enter a question to search.")
        else:
            if qdrant_client is None or collection is None:
                try:
                    qdrant_client, collection = get_qdrant_client(st.session_state.session_id)
                except ValueError as e:
                    st.error(f"Qdrant Client Error during search: {e}. Please check your .env file.")
                    st.stop()

            with st.spinner("Searching across all indexed documents..."):
                st.session_state.search_results = search_qdrant(
                    user_query, embed_model, qdrant_client, collection, k_snippets, similarity_threshold
                )

    # --- RAG Display Results ---
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

# =============================================================================
# --- DOCUMENT EXTRACTION PIPELINE ---
# =============================================================================
elif app_mode == "Document Extraction Pipeline":
    st.title("Document Extraction Pipeline")
    st.markdown("Upload one or more PDFs to extract content as clean Markdown files and download all their images. This uses the local Marker library for high-quality conversion.")

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("1. Document Configuration")
        uploaded_files = st.file_uploader(
            "Upload your documents (PDFs only)",
            type="pdf",
            accept_multiple_files=True,
            key="extraction_uploader"
        )
        page_range_input_extraction = st.text_input(
            "Page Range (e.g., '0,5-10,20'):",
            help="Specify which pages to process (0-indexed). Leave empty for all pages. Changing this will restart processing."
        )
        st.header("2. Process")
        process_button = st.button("✨ Extract Content", type="primary", key="extraction_process_button", disabled=not uploaded_files)

    # Load Moondream and Marker models (with page range config)
    try:
        moondream_model, moondream_limiter = load_moondream_model()
        # Pass page range to model loader. Caching handles re-initialization.
        marker_converter = load_marker_model(page_range_input_extraction)
    except ValueError as e:
        st.error(f"Model Configuration Error: {e}. Please check your .env file for required API keys.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load models. Error: {e}")
        st.stop()

    if process_button and uploaded_files:
        st.session_state.extraction_results = [] # Clear previous results
        with st.spinner(f"Processing {len(uploaded_files)} document(s)... This may take some time."):
            all_results = []
            for uploaded_file in uploaded_files:
                try:
                    st.toast(f"Extracting content from '{uploaded_file.name}'...")
                    file_bytes = uploaded_file.getvalue()
                    # The marker_converter is already pre-configured with the page range.
                    final_md, images_to_save = prepare_document_for_download(
                        file_bytes, uploaded_file.name, moondream_model, moondream_limiter, 
                        marker_converter
                    )
                    basename = os.path.splitext(uploaded_file.name)[0]
                    
                    all_results.append({
                        "filename": uploaded_file.name,
                        "markdown": final_md,
                        "images": images_to_save,
                        "basename": basename
                    })
                except Exception as e:
                    st.error(f"Failed to process '{uploaded_file.name}': {e}")
            st.session_state.extraction_results = all_results
    
    if st.session_state.extraction_results:
        st.subheader("✅ Extraction Complete")
        st.markdown(f"Processed {len(st.session_state.extraction_results)} document(s). View and download the results for each file below.")

        for i, res in enumerate(st.session_state.extraction_results):
            with st.expander(f"Results for: **{res['filename']}** ({len(res['images'])} images found)"):
                
                # Prepare zip file in memory for download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(f"{res['basename']}.md", res['markdown'])
                    for img_name, img_bytes in res['images'].items():
                        zip_file.writestr(f"images/{img_name}", img_bytes)
                
                st.download_button(
                    label=f"⬇️ Download '{res['basename']}.zip'",
                    data=zip_buffer.getvalue(),
                    file_name=f"{res['basename']}.zip",
                    mime="application/zip",
                    key=f"download_{i}" # Unique key for each download button
                )
                
                st.text_area(f"Extracted Markdown for {res['filename']}:", res['markdown'], height=400, key=f"md_preview_{i}")
                st_copy_to_clipboard(res['markdown'], "Copy Markdown to Clipboard", key=f"copy_{i}")