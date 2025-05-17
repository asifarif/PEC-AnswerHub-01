# rag_engine.py
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from data_loader import download_and_load_pdfs
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import tenacity
import hashlib
import json
import re
from rank_bm25 import BM25Okapi

# Disable CUDA to avoid PyTorch issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()

# Validate GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is required.")

# Check PyTorch version
if not torch.__version__.startswith("2."):
    print(f"Warning: PyTorch version {torch.__version__} may not be compatible. Expected 2.x.")

# Use a QA-optimized embedding model
EMBEDDING_MODEL = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
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
        return False

def preprocess_document(text):
    """Preprocess document while preserving numbers, units, and table structures."""
    # Expand references like "Table A & B"
    table_a = "Constructor's Categories: C-A (No limit, 200 PCPs, 2 PEs with 20 years experience, 5 REs as trainees), C-B (Up to 4000 million rupees, 120 PCPs, 2 PEs with 15 years experience, 3 REs as trainees), C-1 (Up to 2500 million rupees, 90 PCPs, 2 PEs with 10 years experience, 2 REs as trainees), C-2 (Up to 1000 million rupees, 35 PCPs, 1 PE and REs, 1 RE as trainee), C-3 (Up to 500 million rupees, 20 PCPs, 50% REs), C-4 (Up to 200 million rupees, 15 PCPs, 50% REs), C-5 (Up to 65 million rupees, 5 PCPs, 1 RE), C-6 (Up to 25 million rupees, 5 PCPs, 1 RE)"
    table_b = "Operator's Categories: O-A (No limit, 150 PCPs, 2 PEs with 20 years experience, 5 REs as trainees), O-B (Up to 2000 million rupees, 100 PCPs, 2 PEs with 15 years experience, 3 REs as trainees), O-1 (Up to 100 million rupees, 75 PCPs, 2 PEs with 10 years experience, 2 REs as trainees), O-2 (Up to 500 million rupees, 35 PCPs, 1 PE and REs, 1 RE as trainee), O-3 (Up to 200 million rupees, 20 PCPs, 50% REs), O-4 (Up to 100 million rupees, 15 PCPs, 50% REs), O-5 (Up to 30 million rupees, 5 PCPs, 1 RE), O-6 (Up to 20 million rupees, 5 PCPs, 1 RE)"
    text = text.replace("Table A & B", f"{table_a}\n{table_b}")
    
    # Preserve numbers, units, and table-related characters
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = re.sub(r'[^\w\s\d.,/%$PKR\-:=|]', ' ', text)  # Keep table delimiters and units
    return text

def chunk_documents(text):
    """Chunk text with larger size to preserve tables, using page breaks if available."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Increased to keep tables intact
        chunk_overlap=300,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", "-"]  # Prioritize paragraph and table breaks
    )
    chunks = text_splitter.split_text(text)
    valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]  # Higher threshold for meaningful content
    return valid_chunks

def create_vector_store():
    """Create and save a FAISS vector store from PEC documents with page metadata."""
    print("Creating vector store...")
    docs = download_and_load_pdfs()
    print(f"Downloaded and loaded {len(docs)} documents: {[doc.metadata.get('source') for doc in docs if isinstance(doc, Document)]}")

    texts, metadatas = [], []
    for doc in docs:
        if not isinstance(doc, Document):
            print(f"Skipping invalid document: {doc}")
            continue

        text = doc.page_content
        metadata = doc.metadata.copy()  # Ensure metadata is not modified
        metadata["page"] = metadata.get("page", 0)  # Default to 0 if page not present
        if not text or len(text.strip()) < 50:
            print(f"Skipping empty or too short document content (page {metadata['page']}): {text[:100]}...")
            continue

        # Preprocess the document content
        processed_text = preprocess_document(text)
        # Create semantic chunks
        chunks = chunk_documents(processed_text)
        if not chunks:
            print(f"No valid chunks generated from content (page {metadata['page']}): {processed_text[:100]}...")
            continue

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                **metadata,
                "chunk": chunk
            })

    print(f"Generated {len(texts)} document chunks")
    if len(texts) == 0:
        raise ValueError("No valid document chunks generated.")

    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    dimension = embeddings.shape[1]
    
    # Use IndexIVFFlat with dynamic nlist
    nlist = min(len(texts), max(1, int(np.sqrt(len(texts)))))
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    print(f"Created IndexIVFFlat with dimension {dimension}, nlist {nlist}")
    
    index.train(np.array(embeddings, dtype=np.float32))
    index.add(np.array(embeddings, dtype=np.float32))
    index.nprobe = max(1, min(nlist, 10))

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump((index, texts, metadatas), f)
    
    current_hash = compute_documents_hash()
    if current_hash:
        with open(HASH_PATH, "wb") as f:
            pickle.dump(current_hash, f)
    
    print("Vector store created and saved.")

def load_vector_store():
    """Load the FAISS vector store, regenerating if invalid."""
    if not check_vector_store_validity():
        print("Vector store invalid or not found. Creating a new one...")
        create_vector_store()
    with open(VECTOR_STORE_PATH, "rb") as f:
        index, texts, metadatas = pickle.load(f)
    print(f"Loaded index type: {type(index).__name__}")
    return index, texts, metadatas

def retrieve_chunks(query, k=25):  # Increased k for more candidates
    """Retrieve top-k relevant document chunks with page-aware hybrid ranking."""
    index, texts, metadatas = load_vector_store()
    query_embedding = EMBEDDING_MODEL.encode([query.lower()])  # Lowercase for consistency
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k * 2)  # Retrieve more candidates
    
    # BM25 keyword search
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Combine FAISS distance and BM25 scores, prioritize page relevance for upgradation
    combined_scores = []
    for j, i in enumerate(I[0]):
        if i < len(bm25_scores):
            score = D[0][j] + (1 - bm25_scores[i]) * 5  # Weight BM25 higher
            if "upgradation" in query.lower() and metadatas[i].get("page") == 13:  # Target Page 13 for upgradation
                score -= 15  # Strong boost for Page 13
            combined_scores.append((i, score))
    
    combined_scores.sort(key=lambda x: x[1])
    top_k_indices = [i for i, _ in combined_scores[:k]]
    
    return [metadatas[i] for i in top_k_indices]

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def query_groq(query):
    """Query the Groq API with retrieved context and metadata."""
    if not query.strip():
        return "Please provide a valid question."

    chunks = retrieve_chunks(query)
    context = "\n\n".join([
        f"Page {c.get('page', 'N/A')}: {c.get('chunk', '')}"
        for c in chunks
    ])

    prompt = f"""
You are an AI assistant for the Pakistan Engineering Council (PEC). Based on the context below, provide a precise answer to the user's question about PEC policies. Focus on extracting specific details like fees, requirements, or duties from tables or text. If the answer involves numbers, include the exact value and unit (e.g., PKR).  

Context:
{context}

User Question: {query}
Answer:
"""

    print("Initializing Groq client...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Grok client initialized successfully.")
    except Exception as e:
        raise

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=700  # Increased for detailed responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise