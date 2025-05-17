import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from data_loader import download_and_load_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH = "vector_store.pkl"
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
#EMBEDDING_MODEL = SentenceTransformer("intfloat/e5-large-v2", device="cpu")


def preprocess_document(text):
    """Preprocess document to preserve units and table structures."""
    return re.sub(r'[^\w\s\d.,/%$PKR\-:=|]', ' ', text)

def chunk_by_sections(text):
    """Chunk text by section headers, including table delimiters."""
    sections = re.split(r'\n(?=[0-9]+\.|[A-Z].+:|\|)', text)  # Add '|' for table columns
    return [s.strip() for s in sections if len(s.strip()) > 100]

def create_vector_store():
    print("Creating vector store...")
    docs = download_and_load_pdfs()
    print(f"Downloaded and loaded {len(docs)} documents: {[doc.metadata.get('title') for doc in docs if isinstance(doc, Document)]}")

    texts, metadatas = [], []
    for doc in docs:
        if not isinstance(doc, Document):
            print(f"Skipping invalid document: {doc}")
            continue

        processed_text = preprocess_document(doc.page_content)
        section_chunks = chunk_by_sections(processed_text)
        if not section_chunks:
            print(f"No valid chunks generated from content (page {doc.metadata.get('page', 0)}): {processed_text[:100]}...")
            continue

        for chunk in section_chunks:
            texts.append(chunk)
            metadatas.append({
                **doc.metadata,
                "chunk": chunk
            })

    print(f"Generated {len(texts)} document chunks")
    if len(texts) == 0:
        raise ValueError("No valid document chunks generated.")

    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump((index, texts, metadatas), f)

def load_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        create_vector_store()
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)

def retrieve_chunks(query, k=15):  # Increased to 15
    index, texts, metadatas = load_vector_store()
    query_embedding = EMBEDDING_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)
    return [metadatas[i] for i in I[0]]

from groq import Groq

def query_groq(query):
    chunks = retrieve_chunks(query)
    context = "\n\n".join([f"Page {c['page']}: {c['chunk']}" for c in chunks])

    prompt = f"""
You are a PEC policy assistant. Based on the context below, extract registration fee tables and answer the user's query clearly.
If the question relates to fees or categories:
- Find and include any structured tables that match.
- Convert them to bullet points or Markdown.  


Context:
{context}

Question: {query}
Answer:
"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()