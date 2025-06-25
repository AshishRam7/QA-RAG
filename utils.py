import os
import uuid
import torch
import streamlit as st
from dotenv import load_dotenv
import tempfile
import base64
import re
from PIL import Image
import io
from collections import deque
from typing import Optional

# Text Processing & Embeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# Moondream
import moondream

# Marker library
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# --- Load Environment Variables and Constants ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
QDRANT_COLLECTION_PREFIX = "streamlit_rag_collection_marker_v2"

# --- Moondream Rate Limiter ---
class MoondreamRateLimiter:
    def __init__(self, calls_per_minute=55):
        self.calls_per_minute = calls_per_minute
        self.last_calls = deque()

    def wait_if_needed(self):
        import time
        now = time.time()
        while self.last_calls and self.last_calls[0] <= now - 60:
            self.last_calls.popleft()
        if len(self.last_calls) >= self.calls_per_minute:
            wait_until = self.last_calls[0] + 60
            sleep_time = wait_until - now
            if sleep_time > 0:
                time.sleep(sleep_time + 0.1)
                self.wait_if_needed()
        self.last_calls.append(time.time())

@st.cache_resource
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Embedding model is running on: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return model

@st.cache_resource
def load_moondream_model():
    if not MOONDREAM_API_KEY:
        raise ValueError("MOONDREAM_API_KEY must be set in the .env file to use features requiring Moondream.")
    st.info("Initializing Moondream model...")
    model = moondream.vl(api_key=MOONDREAM_API_KEY)
    limiter = MoondreamRateLimiter()
    return model, limiter

@st.cache_resource
def load_marker_model(page_range_str: Optional[str] = None):
    """
    Load the local Marker model, configured with an optional page range.
    The page_range_str is used as part of the cache key.
    """
    st.info("Initializing local Marker model...")
    try:
        config_params = {
            "use_llm": True,
            "format_lines": True,
            "redo_inline_math": True,
            "output_format": "markdown"
        }

        # NEW: Add page_range to config if provided
        if page_range_str and page_range_str.strip():
            config_params["page_range"] = page_range_str.strip()
            st.info(f"Marker will process page range: '{config_params['page_range']}'")

        if config_params["use_llm"]:
            if not GEMINI_API_KEY:
                st.warning("GEMINI_API_KEY not set. Marker's LLM features (use_llm, redo_inline_math) will be disabled.")
                config_params["use_llm"] = False
                config_params["redo_inline_math"] = False
            else:
                config_params["gemini_api_key"] = GEMINI_API_KEY
                config_params["llm_service"] = "marker.services.gemini.GoogleGeminiService"
        
        config_parser = ConfigParser(config_params)
        artifact_dict = create_model_dict()
        
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        st.info("Marker model loaded successfully!")
        
        if config_params.get("use_llm") and "llm_service" in config_params:
             st.info(f"Marker LLM features are ENABLED using {config_params['llm_service'].split('.')[-1]}.")
        else:
            st.info("Marker LLM features are DISABLED.")

        return converter
    except Exception as e:
        st.error(f"Failed to load Marker model: {e}")
        raise

@st.cache_resource
def get_qdrant_client(session_id):
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file.")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = f"{QDRANT_COLLECTION_PREFIX}_{session_id}"

    try:
        client.get_collection(collection_name=collection_name)
        st.info(f"Using existing Qdrant collection: `{collection_name}`")
    except Exception:
        st.info(f"Creating new Qdrant collection: `{collection_name}`")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    return client, collection_name

# --- Local Marker Processing Functions ---

def call_local_marker(file_bytes, marker_converter):
    """Use a pre-configured local Marker converter to process a PDF."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            # The converter is already configured with page ranges, etc.
            rendered = marker_converter(temp_file_path)
            markdown_text, metadata, images = text_from_rendered(rendered)
            
            images_b64 = {}
            for img_name, img_pil in images.items():
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images_b64[img_name] = img_b64
            
            return {"markdown": markdown_text, "images": images_b64, "metadata": metadata}
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        raise Exception(f"Local Marker processing failed: {e}")

def get_moondream_description(image_b64, md_model, limiter):
    try:
        limiter.wait_if_needed()
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        encoded_image = md_model.encode_image(image)
        prompt = (
            "Describe the key technical findings in good technical detail in this figure/visualization. "
            "Illustrate and mention trends, patterns, and numerical values that can be observed. Provide a scientific/academic styled short, "
            "single paragraph summary that is highly insightful in context of the document."
        )
        response = md_model.query(encoded_image, prompt)
        
        if isinstance(response, dict):
            return response.get("answer", response.get("text", "")).strip()
        elif isinstance(response, str):
            return response.strip()
        return f"Unexpected response type from Moondream: {type(response)}"
    except Exception as e:
        return f"Error in Moondream processing: {str(e)}"

def enrich_markdown_for_rag(markdown_text, image_descriptions):
    def replace_func(match):
        image_path = match.group(2)
        description = image_descriptions.get(image_path, "No description generated.")
        return f"\n\n--- Image Description ---\n{description}\n--- End Image Description ---\n\n"

    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    return image_pattern.sub(replace_func, markdown_text)

def chunk_text_with_langchain(text):
    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    return markdown_splitter.split_text(text)

# --- RAG Pipeline Function ---

def process_and_embed_document(uploaded_file, embed_model, md_model, md_limiter, marker_converter, qdrant_client, collection_name):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner(f"1/4: Converting '{filename}' using Marker..."):
        try:
            marker_result = call_local_marker(file_bytes, marker_converter)
            raw_md = marker_result.get("markdown", "")
            images_b64 = marker_result.get("images", {})
            st.success(f"Marker processing complete for '{filename}'. Found {len(images_b64)} images.")
        except Exception as e:
            st.error(f"Failed during Marker processing for {filename}: {e}")
            return False

    with st.spinner(f"2/4: Generating descriptions for {len(images_b64)} images..."):
        image_descriptions = {}
        if images_b64:
            progress_bar = st.progress(0, text=f"Describing images in {filename}...")
            for i, (img_path, img_b64) in enumerate(images_b64.items()):
                desc = get_moondream_description(img_b64, md_model, md_limiter)
                image_descriptions[img_path] = desc
                progress_bar.progress((i + 1) / len(images_b64), text=f"Describing image {i+1}/{len(images_b64)}")
            progress_bar.empty()
            st.success(f"Moondream descriptions generated for '{filename}'.")
        else:
            st.info(f"No images to describe in '{filename}'.")

    with st.spinner(f"3/4: Enriching and chunking text for '{filename}'..."):
        enriched_md = enrich_markdown_for_rag(raw_md, image_descriptions)
        text_chunks = chunk_text_with_langchain(enriched_md)
        if not text_chunks:
            st.warning(f"No text chunks were generated from {filename}. Skipping.")
            return False
        st.success(f"Created {len(text_chunks)} text chunks for '{filename}'.")

    with st.spinner(f"4/4: Embedding and indexing chunks for '{filename}'..."):
        embeddings = embed_model.encode(text_chunks, show_progress_bar=True)
        points_to_upsert = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk, "source_file": filename, "session_id": collection_name.split('_')[-1]} 
            ) for chunk, emb in zip(text_chunks, embeddings)
        ]
        try:
            qdrant_client.upsert(collection_name=collection_name, points=points_to_upsert, wait=True, batch_size=64)
            st.success(f"Successfully indexed '{filename}' with {len(points_to_upsert)} snippets.")
            return True
        except Exception as e:
            st.error(f"Failed to upsert data to Qdrant for {filename}: {e}")
            return False

# --- Document Extraction Pipeline Function ---

def prepare_document_for_download(file_bytes, filename, md_model, md_limiter, marker_converter):
    marker_result = call_local_marker(file_bytes, marker_converter)
    raw_md = marker_result.get("markdown", "")
    images_b64 = marker_result.get("images", {})

    image_descriptions = {}
    if images_b64:
        st.write(f"Generating descriptions for {len(images_b64)} images...")
        image_progress_bar = st.progress(0, text=f"Describing images in {filename}...")
        for i, (img_path, img_b64) in enumerate(images_b64.items()):
            desc = get_moondream_description(img_b64, md_model, md_limiter)
            image_descriptions[img_path] = desc
            image_progress_bar.progress((i + 1) / len(images_b64), text=f"Describing image {i+1}/{len(images_b64)}")
        image_progress_bar.empty()
        st.success(f"Moondream descriptions generated for '{filename}'.")

    md_with_descriptions = enrich_markdown_for_rag(raw_md, image_descriptions)

    images_to_save = {}
    for i, (marker_path, img_b64_data) in enumerate(images_b64.items()):
        extension = os.path.splitext(marker_path)[1] or '.png'
        new_filename = f"image_{i}{extension}"
        images_to_save[new_filename] = base64.b64decode(img_b64_data)

    return md_with_descriptions, images_to_save

# --- Search and Formatting Functions ---

def search_qdrant(query, model, qdrant_client, collection_name, k, similarity_threshold):
    if not query:
        return []
    query_embedding = model.encode(query).tolist()
    
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k,
        score_threshold=similarity_threshold
    )

def create_passage_from_snippets(search_results):
    if not search_results:
        return "No relevant snippets were found to construct a passage."
    return "\n\n---\n\n".join([result.payload.get("text", "") for result in search_results])