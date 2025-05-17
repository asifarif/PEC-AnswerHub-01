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

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_STORE_PATH = "vector_store.pkl"
GROQ_MODEL = "llama3-70b-8192"

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

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                **metadata,
                "chunk": chunk
            })

    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump((index, texts, metadatas), f)
    logger.info("Vector store created and saved.")

def load_vector_store():
    """Load the FAISS vector store."""
    if not os.path.exists(VECTOR_STORE_PATH):
        logger.warning("Vector store not found. Creating a new one...")
        create_vector_store()
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)

def retrieve_chunks(query, k=5):
    """Retrieve top-k relevant document chunks for the query."""
    index, texts, metadatas = load_vector_store()
    query_embedding = EMBEDDING_MODEL.encode([query])
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
You are an AI assistant for the Pakistan Engineering Council (PEC). Based on the context below, provide a precise answer to the user's question about PEC registration policies. Include relevant policy title, policy number, and approval date if available.

Context:
{context}

User Question: {query}
Answer:
"""

    client = Groq(api_key=GROQ_API_KEY)
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
    