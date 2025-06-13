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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Embedding model is running on: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return model

@st.cache_resource
def load_moondream_model():
    if not MOONDREAM_API_KEY:
        raise ValueError("MOONDREAM_API_KEY must be set in the .env file.")
    st.info("Initializing Moondream model...")
    model = moondream.vl(api_key=MOONDREAM_API_KEY)
    return model

@st.cache_resource
def get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file.")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception:
        st.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating it...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    return client


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
    """
    FIXED: Generates a description for a base64 encoded image, correctly handling the dictionary response.
    """
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB": image = image.convert("RGB")
        
        encoded_image = md_model.encode_image(image)
        prompt = "Describe the key information, data, or technical findings in this image. Focus on content relevant for analysis."
        
        response = md_model.query(encoded_image, prompt)
        
        description = ""
        # The API returns a dictionary. We need to extract the text from it.
        if isinstance(response, dict):
            description = response.get("answer", response.get("text", "Could not find text in Moondream response."))
        elif isinstance(response, str):
            description = response
        else:
            description = f"Unexpected response type from Moondream: {type(response)}"
            
        return description.strip()
    except Exception as e:
        return f"Error in Moondream processing: {str(e)}"

def enrich_markdown(markdown_text, image_descriptions):
    def replace_func(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        description = image_descriptions.get(image_path, "No description generated.")
        return f"{description}"

    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    enriched_text = image_pattern.sub(replace_func, markdown_text)
    return enriched_text

def chunk_text_with_langchain(text):
    """
    FIXED: Increased chunk size to create more substantial, meaningful chunks.
    """
    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], # Added ". " as a separator
        chunk_size=2000,   # Increased from 1000
        chunk_overlap=200,    # Increased from 150
        length_function=len,
    )
    return markdown_splitter.split_text(text)


def process_and_embed_document(uploaded_file, embed_model, md_model, qdrant_client, session_id):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner(f"1/4: Sending '{filename}' to Marker API..."):
        try:
            marker_result = call_datalab_marker(file_bytes, filename)
            raw_md = marker_result.get("markdown", "")
            images_b64 = marker_result.get("images", {})
            st.success(f"Marker processing complete. Found {len(images_b64)} images.")
        except Exception as e:
            st.error(f"Failed during Marker processing for {filename}: {e}")
            return False

    with st.spinner(f"2/4: Generating descriptions for {len(images_b64)} images with Moondream..."):
        image_descriptions = {}
        if images_b64:
            progress_bar = st.progress(0, text="Describing images...")
            for i, (img_path, img_b64) in enumerate(images_b64.items()):
                desc = get_moondream_description(img_b64, md_model)
                image_descriptions[img_path] = desc
                progress_bar.progress((i + 1) / len(images_b64))
            progress_bar.empty()
            st.success("Moondream image descriptions generated.")
        else:
            st.info("No images found to describe.")

    with st.spinner("3/4: Enriching markdown and creating intelligent text chunks..."):
        enriched_md = enrich_markdown(raw_md, image_descriptions)
        text_chunks = chunk_text_with_langchain(enriched_md)
        if not text_chunks:
            st.warning(f"No text chunks were generated from {filename}. Skipping.")
            return False
        st.success(f"Created {len(text_chunks)} intelligent text chunks.")

    with st.spinner(f"4/4: Embedding chunks and upserting to Qdrant..."):
        embeddings = embed_model.encode(text_chunks, show_progress_bar=True)
        points_to_upsert = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk, "session_id": session_id, "source_file": filename}
            ) for chunk, emb in zip(text_chunks, embeddings)
        ]
        try:
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_upsert, wait=True)
            st.success(f"Successfully indexed {filename} with {len(points_to_upsert)} enriched snippets.")
            return True
        except Exception as e:
            st.error(f"Failed to upsert data to Qdrant for {filename}: {e}")
            return False

def search_qdrant(query, model, qdrant_client, session_id, k, similarity_threshold):
    if not query: return []
    try:
        query_embedding = model.encode(query)
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]),
            limit=k, score_threshold=similarity_threshold
        )
        return search_results
    except Exception as e:
        st.error(f"An error occurred during Qdrant search: {e}")
        return []

def create_passage_from_snippets(search_results):
    if not search_results:
        return "No relevant snippets were found to construct a passage."
    return "\n\n---\n\n".join([result.payload.get("text", "") for result in search_results])