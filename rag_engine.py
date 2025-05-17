import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from data_loader import download_and_load_pdfs
from langchain.docstore.document import Document
import logging
import torch
import tenacity
import hashlib
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable CUDA to avoid PyTorch issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()

# Validate GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables.")
    raise ValueError("GROQ_API_KEY is required.")

# Check PyTorch version
if not torch.__version__.startswith("2."):
    logger.warning(f"PyTorch version {torch.__version__} may not be compatible. Expected 2.x.")

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L12-v2")
VECTOR_STORE_PATH = "vector_store.pkl"
GROQ_MODEL = "llama3-70b-8192"
DOCUMENTS_DIR = "documents"
POLICY_LINKS_PATH = "policy_links.json"
HASH_PATH = "vector_store_hash.pkl"

def compute_file_hash(file_path):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_documents_hash():
    """Compute hash of policy_links.json and all PDFs in documents/."""
    try:
        with open(POLICY_LINKS_PATH, "r") as f:
            policy_data = json.load(f)
        policy_hash = hashlib.sha256(json.dumps(policy_data, sort_keys=True).encode()).hexdigest()
        
        pdf_hashes = []
        for item in policy_data.get("policies", []):
            pdf_path = os.path.join(DOCUMENTS_DIR, f"{item['title']}.pdf")
            if os.path.exists(pdf_path):
                pdf_hashes.append(compute_file_hash(pdf_path))
        
        combined = policy_hash + "".join(sorted(pdf_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Error computing documents hash: {str(e)}")
        return None

def check_vector_store_validity():
    """Check if vector_store.pkl is valid by comparing file hashes."""
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(HASH_PATH):
        return False
    
    current_hash = compute_documents_hash()
    if not current_hash:
        return False
    
    try:
        with open(HASH_PATH, "rb") as f:
            stored_hash = pickle.load(f)
        return current_hash == stored_hash
    except Exception as e:
        logger.error(f"Error checking vector store hash: {str(e)}")
        return False

def create_vector_store():
    """Create and save a FAISS vector store from PEC documents."""
    logger.info("Creating vector store...")
    docs = download_and_load_pdfs()

    texts, metadatas = [], []
    for doc in docs:
        if not isinstance(doc, Document):
            continue

        text = doc.page_content
        metadata = doc.metadata

        # Create overlapping chunks (1000 chars, 200-char overlap)
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Ignore very small chunks
                chunks.append(chunk)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                **metadata,
                "chunk": chunk
            })

    logger.info(f"Generated {len(texts)} document chunks")
    if len(texts) == 0:
        logger.error("No valid document chunks generated.")
        raise ValueError("No valid document chunks generated.")

    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    
    # Use IndexIVFFlat with dynamic nlist
    nlist = min(10, len(texts))  # Ensure nlist <= number of chunks
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    logger.info(f"Created IndexIVFFlat with dimension {dimension}, nlist {nlist}")
    
    # Train index only if sufficient data
    if len(embeddings) < nlist:
        logger.error(f"Insufficient chunks ({len(embeddings)}) for nlist ({nlist})")
        raise ValueError(f"Number of chunks ({len(embeddings)}) must be at least {nlist}")
    
    index.train(np.array(embeddings, dtype=np.float32))
    index.add(np.array(embeddings, dtype=np.float32))
    index.nprobe = 5  # Reduced for smaller nlist

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump((index, texts, metadatas), f)
    
    # Save hash of current documents
    current_hash = compute_documents_hash()
    if current_hash:
        with open(HASH_PATH, "wb") as f:
            pickle.dump(current_hash, f)
    
    logger.info("Vector store created and saved.")

def load_vector_store():
    """Load the FAISS vector store, regenerating if invalid."""
    if not check_vector_store_validity():
        logger.warning("Vector store invalid or not found. Creating a new one...")
        create_vector_store()
    with open(VECTOR_STORE_PATH, "rb") as f:
        index, texts, metadatas = pickle.load(f)
    logger.info(f"Loaded index type: {type(index).__name__}")
    return index, texts, metadatas

def retrieve_chunks(query, k=10):
    """Retrieve top-k relevant document chunks for the query."""
    index, texts, metadatas = load_vector_store()
    query_embedding = EMBEDDING_MODEL.encode([query])
    logger.info(f"Searching index type: {type(index).__name__}, k={k}")
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)
    return [metadatas[i] for i in I[0]]

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def query_groq(query):
    """Query the Grok API with retrieved context."""
    if not query.strip():
        return "Please provide a valid question."

    chunks = retrieve_chunks(query)
    context = "\n\n".join([
        f"Title: {c.get('title')}\nPolicy Number: {c.get('policy_number')}\nApproval Date: {c.get('approval_date')}\nContent: {c.get('chunk')}"
        for c in chunks
    ])

    prompt = f"""
You are an AI assistant for the Pakistan Engineering Council (PEC). Based on the context below, provide a precise answer to the user's question about PEC policies. 

Context:
{context}

User Question: {query}
Answer:
"""

    logger.info("Initializing Groq client...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Grok client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        raise

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error querying Grok API: {str(e)}")
        raise