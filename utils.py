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
from collections import deque # Added for rate limiting

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
from marker.config.parser import ConfigParser # NEW IMPORT

# --- Load Environment Variables and Constants ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # NEW ENV VAR for Marker's LLM features

EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
# Base collection name prefix; a session ID will be appended to it
QDRANT_COLLECTION_PREFIX = "streamlit_rag_collection_marker_v2" 

# --- Moondream Rate Limiter ---
class MoondreamRateLimiter:
    """
    Limits calls to Moondream API to a specified rate (e.g., 60 calls per minute).
    Uses a deque to track call timestamps and sleeps if the rate limit is approached.
    """
    def __init__(self, calls_per_minute=55): # Set slightly below 60 for safety
        self.calls_per_minute = calls_per_minute
        self.last_calls = deque() # Stores timestamps of last calls

    def wait_if_needed(self):
        import time
        now = time.time()
        
        # Remove calls older than 60 seconds
        while self.last_calls and self.last_calls[0] <= now - 60:
            self.last_calls.popleft()

        # If we've made too many calls recently, wait
        if len(self.last_calls) >= self.calls_per_minute:
            # Calculate when the earliest call in the current window will expire
            wait_until = self.last_calls[0] + 60
            sleep_time = wait_until - now
            if sleep_time > 0:
                # print(f"Rate limit hit. Sleeping for {sleep_time:.2f} seconds.") # For debugging
                time.sleep(sleep_time + 0.1) # Add a small buffer
                self.wait_if_needed() # Re-check after sleeping (recursive call to ensure limit is met)

        self.last_calls.append(time.time()) # Record this call


@st.cache_resource
def load_embedding_model():
    # Required only for RAG Search tool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Embedding model is running on: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return model

@st.cache_resource
def load_moondream_model():
    # Required for both RAG Search and Document Extraction tools
    if not MOONDREAM_API_KEY:
        raise ValueError("MOONDREAM_API_KEY must be set in the .env file to use features requiring Moondream.")
    st.info("Initializing Moondream model...")
    model = moondream.vl(api_key=MOONDREAM_API_KEY)
    limiter = MoondreamRateLimiter() # Instantiate the rate limiter
    return model, limiter # Return both the model and the limiter

@st.cache_resource
def load_marker_model():
    """Load the local Marker model for PDF conversion with advanced settings."""
    st.info("Initializing local Marker model with LLM-enhanced features...")
    try:
        # Define the configuration for Marker based on the user's guide
        config_params = {
            "use_llm": True,  # Enable LLM for higher quality processing
            "format_lines": True,  # Format lines using OCR model (for better math, underlines, bold, etc.)
            "redo_inline_math": True,  # For highest quality inline math conversion (works with use_llm)
            "output_format": "markdown"
        }

        # Conditionally set LLM service based on availability of GEMINI_API_KEY
        if config_params["use_llm"]:
            if not GEMINI_API_KEY:
                st.warning("GEMINI_API_KEY is not set. Marker's LLM features (use_llm, redo_inline_math) will be disabled.")
                config_params["use_llm"] = False
                config_params["redo_inline_math"] = False # This depends on use_llm
                config_params["output_format"] = "markdown"
            else:
                # Default LLM service to Google Gemini as per Marker's documentation
                config_params["gemini_api_key"] = GEMINI_API_KEY
                config_params["llm_service"] = "marker.services.gemini.GoogleGeminiService"
                config_params["output_format"] = "markdown"
                # Marker's GoogleGeminiService is designed to pick up GEMINI_API_KEY or GOOGLE_API_KEY from environment variables.
        
        config_parser = ConfigParser(config_params)
        artifact_dict = create_model_dict()
        
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() # This will be None if use_llm is False or service not configured
        )
        st.info("Marker model loaded successfully with specified configurations!")
        
        if config_params["use_llm"] and "llm_service" in config_params:
             st.info(f"Marker LLM features are ENABLED using {config_params['llm_service'].split('.')[-1]}.")
        else:
            st.info("Marker LLM features are DISABLED (either use_llm is False or GEMINI_API_KEY is missing).")

        return converter
    except Exception as e:
        st.error(f"Failed to load Marker model: {e}")
        raise ValueError(f"Failed to load Marker model: {e}")

@st.cache_resource
def get_qdrant_client(session_id):
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the .env file to use the RAG tool.")
    
    # Add check_compatibility=False to avoid client/server version warnings
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    # Instead of a single global name, append the session:
    collection_name = f"{QDRANT_COLLECTION_PREFIX}_{session_id}"

    try:
        # Check if collection exists
        client.get_collection(collection_name=collection_name)
        st.info(f"Using existing Qdrant collection: `{collection_name}`")
    except Exception:
        # If not, create it
        st.info(f"Creating new Qdrant collection: `{collection_name}`")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
    return client, collection_name


# --- Local Marker Processing Functions ---

def call_local_marker(file_bytes, filename, marker_converter):
    """Use local Marker library to convert PDF to markdown and extract images"""
    try:
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Convert using local Marker
            rendered = marker_converter(temp_file_path)
            
            # Extract text, metadata, and images using Marker's output functions
            markdown_text, metadata, images = text_from_rendered(rendered)
            
            # Convert images to base64 format for compatibility with existing code
            images_b64 = {}
            for img_name, img_pil in images.items():
                # Convert PIL image to base64
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images_b64[img_name] = img_b64
            
            return {
                "markdown": markdown_text,
                "images": images_b64,
                "metadata": metadata
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise Exception(f"Local Marker processing failed: {e}")

def get_moondream_description(image_b64, md_model, limiter):
    """Generates a description for a base64 encoded image, respecting rate limits."""
    try:
        limiter.wait_if_needed() # Wait before making the API call
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB": image = image.convert("RGB")
        
        encoded_image = md_model.encode_image(image)
        prompt = (
            f"Describe the key technical findings in good technical detail inthis figure/visualization "
            f"Illustrate and mention trends, "
            f"patterns, and numerical values that can be observed. Provide a scientific/academic styled short, "
            f"single paragraph summary that is highly insightful in context of the document."
        )
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
    """Replaces image links with their AI-generated descriptions for RAG context or output."""
    def replace_func(match):
        image_path = match.group(2)
        description = image_descriptions.get(image_path, "No description generated.")
        # Ensure the description is wrapped to clearly distinguish it
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

def process_and_embed_document(uploaded_file, embed_model, md_model, md_limiter, marker_converter, qdrant_client, collection_name):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner(f"1/4: Converting '{filename}' using local Marker..."):
        try:
            marker_result = call_local_marker(file_bytes, filename, marker_converter)
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
                # Pass the limiter to get_moondream_description
                desc = get_moondream_description(img_b64, md_model, md_limiter)
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
                # Store the actual collection name in payload if needed, or derived session_id
                payload={"text": chunk, "source_file": filename, "session_id": collection_name.split('_')[-1]} 
            ) for chunk, emb in zip(text_chunks, embeddings)
        ]
        batch_size = 64
        try:
            for i in range(0, len(points_to_upsert), batch_size):
                batch = points_to_upsert[i:i+batch_size]
                # Use the provided collection_name
                qdrant_client.upsert(collection_name=collection_name, points=batch, wait=True)

            st.success(f"Successfully indexed '{filename}' with {len(points_to_upsert)} enriched snippets.")
            return True
        except Exception as e:
            st.error(f"Failed to upsert data to Qdrant for {filename}: {e}")
            return False

# --- Document Extraction Pipeline Function ---

def prepare_document_for_download(file_bytes, filename, md_model, md_limiter, marker_converter):
    """
    Processes a PDF file to extract markdown and images using local Marker, including Moondream descriptions
    in the markdown output, and prepares images for download.
    """
    # 1. Call local Marker to get raw markdown and base64 encoded images
    marker_result = call_local_marker(file_bytes, filename, marker_converter)
    raw_md = marker_result.get("markdown", "")
    images_b64 = marker_result.get("images", {})

    # 2. Generate Moondream descriptions for images
    image_descriptions = {}
    if images_b64:
        st.write(f"Generating descriptions for {len(images_b64)} images...")
        image_progress_bar = st.progress(0, text=f"Describing images in {filename}...")
        for i, (img_path, img_b64) in enumerate(images_b64.items()):
            desc = get_moondream_description(img_b64, md_model, md_limiter)
            image_descriptions[img_path] = desc
            image_progress_bar.progress((i + 1) / len(images_b64), text=f"Describing image {i+1}/{len(images_b64)} in {filename}")
        image_progress_bar.empty()
        st.success(f"Moondream descriptions generated for '{filename}'.")
    else:
        st.info(f"No images found to describe in '{filename}'.")

    # 3. Enrich markdown with image descriptions
    md_with_descriptions = enrich_markdown_for_rag(raw_md, image_descriptions)

    # 4. Prepare original image files for download (renaming paths for local structure)
    images_to_save = {}
    
    for i, (marker_path, img_b64_data) in enumerate(images_b64.items()):
        extension = os.path.splitext(marker_path)[1] or '.png'
        new_filename = f"image_{i}{extension}"
        # Store decoded image bytes for zipping
        images_to_save[new_filename] = base64.b64decode(img_b64_data)

    return md_with_descriptions, images_to_save


# --- Search and Formatting Functions ---


def search_qdrant(query, model, qdrant_client, collection_name, k, similarity_threshold):

    if not query:
        return []
    # 1. Encode query
    query_embedding = model.encode(query).tolist()
    
    raw_results = qdrant_client.search(
        collection_name=collection_name, # Use the session-specific collection
        query_vector=query_embedding,
        limit=k,
        score_threshold=similarity_threshold
    )
    
    # Since the collection is session-specific, all results from this collection
    # are relevant to the current session. No further filtering by 'session_id' payload is needed.
    return raw_results



def create_passage_from_snippets(search_results):
    if not search_results:
        return "No relevant snippets were found to construct a passage."
    return "\n\n---\n\n".join([result.payload.get("text", "") for result in search_results])