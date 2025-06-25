import os
import uuid
import torch
import streamlit as st
from dotenv import load_dotenv
import requests
import time
import base64
import re
from PIL import Image
import io

# Text Processing & Embeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Moondream
import moondream

# --- Load Environment Variables and Constants ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATALAB_MARKER_URL = os.getenv("DATALAB_MARKER_URL")
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")

EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
QDRANT_COLLECTION_NAME = "streamlit_rag_collection_marker_v2"
DATALAB_POLL_INTERVAL = 5 # seconds
DATALAB_MAX_POLLS = 120 # 10 minutes max


# --- Model and Client Loading (Cached) ---

@st.cache_resource
def load_embedding_model():
    # Required only for RAG Search tool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Embedding model is running on: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return model

@st.cache_resource
def load_moondream_model():
    # Required only for RAG Search tool
    if not MOONDREAM_API_KEY:
        raise ValueError("MOONDREAM_API_KEY must be set in the .env file to use the RAG tool.")
    st.info("Initializing Moondream model...")
    model = moondream.vl(api_key=MOONDREAM_API_KEY)
    return model

@st.cache_resource
def get_qdrant_client(session_id):
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file to use the RAG tool.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Instead of a single global name, append the session:
    collection_name = f"{QDRANT_COLLECTION_NAME}_{session_id}"

    try:
        client.get_collection(collection_name=collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )

    return client, collection_name



# --- API Call and Processing Functions ---

def call_datalab_marker(file_bytes, filename):
    if not DATALAB_MARKER_URL or not DATALAB_API_KEY:
        raise ValueError("DATALAB_MARKER_URL and DATALAB_API_KEY must be set.")
    headers = {"X-Api-Key": DATALAB_API_KEY}
    files = {"file": (filename, file_bytes, "application/pdf")}
    
    try:
        response = requests.post(DATALAB_MARKER_URL, headers=headers, files=files, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Datalab API error: {data.get('error', 'Unknown error')}")
        check_url = data["request_check_url"]
    except requests.RequestException as e:
        raise Exception(f"Failed to call Datalab Marker API: {e}")

    for _ in range(DATALAB_MAX_POLLS):
        time.sleep(DATALAB_POLL_INTERVAL)
        try:
            poll_resp = requests.get(check_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            if poll_data.get("status") == "complete":
                return {"markdown": poll_data.get("markdown", ""), "images": poll_data.get("images", {})}
            if poll_data.get("status") == "error":
                raise Exception(f"Datalab processing failed: {poll_data.get('error', 'Unknown error')}")
        except requests.RequestException as e:
            st.warning(f"Polling Datalab failed: {e}. Retrying...")
    raise TimeoutError("Polling timed out for Datalab Marker processing.")

def get_moondream_description(image_b64, md_model):
    """Generates a description for a base64 encoded image."""
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB": image = image.convert("RGB")
        
        encoded_image = md_model.encode_image(image)
        prompt = "Describe the key information, data, or technical findings in this image. Focus on content relevant for analysis."
        
        response = md_model.query(encoded_image, prompt)
        
        description = ""
        if isinstance(response, dict):
            description = response.get("answer", response.get("text", "Could not find text in Moondream response."))
        elif isinstance(response, str):
            description = response
        else:
            description = f"Unexpected response type from Moondream: {type(response)}"
            
        return description.strip()
    except Exception as e:
        return f"Error in Moondream processing: {str(e)}"

def enrich_markdown_for_rag(markdown_text, image_descriptions):
    """Replaces image links with their AI-generated descriptions for RAG context."""
    def replace_func(match):
        image_path = match.group(2)
        description = image_descriptions.get(image_path, "No description generated.")
        return f"\n\n--- Image Description ---\n{description}\n--- End Image Description ---\n\n"

    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    enriched_text = image_pattern.sub(replace_func, markdown_text)
    return enriched_text

def chunk_text_with_langchain(text):
    """Splits text into context-aware chunks for embedding."""
    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    return markdown_splitter.split_text(text)

# --- RAG Pipeline Function ---

def process_and_embed_document(uploaded_file, embed_model, md_model, qdrant_client, session_id):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner(f"1/4: Sending '{filename}' to Marker API..."):
        try:
            marker_result = call_datalab_marker(file_bytes, filename)
            raw_md = marker_result.get("markdown", "")
            images_b64 = marker_result.get("images", {})
            st.success(f"Marker processing complete for '{filename}'. Found {len(images_b64)} images.")
        except Exception as e:
            st.error(f"Failed during Marker processing for {filename}: {e}")
            return False

    with st.spinner(f"2/4: Generating descriptions for {len(images_b64)} images in '{filename}'..."):
        image_descriptions = {}
        if images_b64:
            progress_bar = st.progress(0, text=f"Describing images in {filename}...")
            for i, (img_path, img_b64) in enumerate(images_b64.items()):
                desc = get_moondream_description(img_b64, md_model)
                image_descriptions[img_path] = desc
                progress_bar.progress((i + 1) / len(images_b64), text=f"Describing image {i+1}/{len(images_b64)} in {filename}")
            progress_bar.empty()
            st.success(f"Moondream descriptions generated for '{filename}'.")
        else:
            st.info(f"No images found to describe in '{filename}'.")

    with st.spinner(f"3/4: Enriching markdown and chunking text for '{filename}'..."):
        enriched_md = enrich_markdown_for_rag(raw_md, image_descriptions)
        text_chunks = chunk_text_with_langchain(enriched_md)
        if not text_chunks:
            st.warning(f"No text chunks were generated from {filename}. Skipping.")
            return False
        st.success(f"Created {len(text_chunks)} intelligent text chunks for '{filename}'.")

    with st.spinner(f"4/4: Embedding and upserting chunks for '{filename}'..."):
        embeddings = embed_model.encode(text_chunks, show_progress_bar=True)
        points_to_upsert = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk, "session_id": session_id, "source_file": filename}
            ) for chunk, emb in zip(text_chunks, embeddings)
        ]
        batch_size = 64
        try:
            for i in range(0, len(points_to_upsert), batch_size):
                batch = points_to_upsert[i:i+batch_size]
                qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch, wait=True)
            st.success(f"Successfully indexed '{filename}' with {len(points_to_upsert)} enriched snippets.")
            return True
        except Exception as e:
            st.error(f"Failed to upsert data to Qdrant for {filename}: {e}")
            return False

# --- Document Extraction Pipeline Function ---

def prepare_document_for_download(file_bytes, filename):
    """
    Processes a PDF file to extract markdown and images. This version does NOT
    generate AI captions and is used for the Document Extraction Pipeline.
    """
    # 1. Call Marker API to get raw markdown and base64 encoded images
    marker_result = call_datalab_marker(file_bytes, filename)
    raw_md = marker_result.get("markdown", "")
    images_b64 = marker_result.get("images", {})

    # 2. Prepare final markdown and image files for download
    images_to_save = {}
    path_mapping = {}
    
    # Create new filenames and decode image data
    for i, (marker_path, img_b64_data) in enumerate(images_b64.items()):
        # Use original extension if available, otherwise default to .png
        extension = os.path.splitext(marker_path)[1] or '.png'
        new_filename = f"image_{i}{extension}"
        path_mapping[marker_path] = new_filename
        # Store decoded image bytes for zipping
        images_to_save[new_filename] = base64.b64decode(img_b64_data)

    def replace_image_paths_for_download(match):
        marker_path = match.group(2) # The original path from marker
        
        if marker_path in path_mapping:
            alt_text = match.group(1)
            new_path = path_mapping[marker_path]
            # Format for local viewing: ![alt text](images/new_filename.png)
            return f"![{alt_text}](images/{new_path})"
        return match.group(0) # Return original if no match found

    # Use regex to find and replace all image links
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    final_md = image_pattern.sub(replace_image_paths_for_download, raw_md)

    return final_md, images_to_save


# --- Search and Formatting Functions ---

def search_qdrant(query, model, qdrant_client, session_id, k, similarity_threshold):
    if not query:
        return []
    # 1. Encode query
    query_embedding = model.encode(query).tolist()
    # 2. Pull back top‑k from Qdrant without any filter
    raw_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=k,
        score_threshold=similarity_threshold
    )
    # 3. Keep only results for THIS session
    filtered = [
        res for res in raw_results
        if res.payload.get("session_id") == session_id
    ]
    return filtered


def create_passage_from_snippets(search_results):
    if not search_results:
        return "No relevant snippets were found to construct a passage."
    return "\n\n---\n\n".join([result.payload.get("text", "") for result in search_results])