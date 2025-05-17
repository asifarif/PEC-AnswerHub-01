# data_loader.py
import os
import json
import gdown
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "documents"

def download_and_load_pdfs(json_path="policy_links.json"):
    """Download PEC PDFs from Google Drive and load them as Documents."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "policies" not in data or not isinstance(data["policies"], list):
            logger.error(f"Invalid JSON structure in {json_path}.")
            return []
    except FileNotFoundError:
        logger.error(f"JSON file {json_path} not found.")
        return []

    documents = []
    for item in data["policies"]:
        title = item["title"]
        gdrive_id = item["gdrive_id"]
        filename = f"{DOCUMENTS_DIR}/{title}.pdf"

        if not os.path.exists(filename):
            logger.info(f"Downloading: {title}")
            try:
                gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", filename, quiet=False)
            except Exception as e:
                logger.error(f"Failed to download {title}: {str(e)}")
                continue

        try:
            reader = PdfReader(filename)
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            
            doc = Document(
                page_content=full_text,
                metadata={
                    "title": title,
                    "approval_date": item.get("approval_date", "Unknown"),
                    "policy_number": item.get("policy_number", "N/A")
                }
            )
            documents.append(doc)
            logger.info(f"Loaded document: {title}")
        except Exception as e:
            logger.error(f"Failed to process {title}: {str(e)}")
            continue

    if not documents:
        logger.warning("No documents loaded.")
    return documents
