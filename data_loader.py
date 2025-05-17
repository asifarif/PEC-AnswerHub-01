import json
import requests
import pdfplumber
import io
from langchain.docstore.document import Document
import logging

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
)
logger = logging.getLogger(__name__)

def download_pdf_from_google_drive(file_id):
    """Download a PDF from Google Drive using file ID and return bytes."""
    try:
        URL = "https://drive.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # Handle Google Drive's confirmation page for large files
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
                break
        
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Failed to download file {file_id}: Status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {str(e)}")
        return None

def download_and_load_pdfs(json_path="policy_links.json"):
    """Load PDF documents from Google Drive links specified in JSON."""
    documents = []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "policies" not in data:
            logger.error("Invalid JSON structure: 'policies' key not found.")
            return []
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return []

    for item in data["policies"]:
        title = item.get("title")
        gdrive_id = item.get("gdrive_id")

        if not title or not gdrive_id:
            logger.error(f"Skipping invalid entry: {item}")
            continue

        # Download PDF
        pdf_bytes = download_pdf_from_google_drive(gdrive_id)
        if not pdf_bytes:
            logger.error(f"Failed to download PDF for {title}")
            continue

        try:
            # Process PDF in-memory
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    table_text = ""
                    tables = page.find_tables()

                    for table in tables:
                        extracted = table.extract()
                        if extracted:
                            table_text += "\n".join([
                                " | ".join(cell if cell is not None else "" for cell in row)
                                for row in extracted if any(row)
                            ]) + "\n"

                    combined_text = text + "\n" + table_text

                    if len(combined_text.strip()) < 50:
                        continue

                    documents.append(Document(
                        page_content=combined_text.strip(),
                        metadata={
                            "title": title,
                            "approval_date": item.get("approval_date", "Unknown"),
                            "policy_number": item.get("policy_number", "N/A"),
                            "page": i + 1
                        }
                    ))
        except Exception as e:
            logger.error(f"Failed to extract {title}: {str(e)}")

    return documents