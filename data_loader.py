# data_loader.py
import os
import json
import pdfplumber
from langchain.docstore.document import Document
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "documents"

def download_and_load_pdfs(json_path="policy_links.json"):
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "policies" not in data:
            logger.error("Invalid JSON structure: 'policies' key not found.")
            return []
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return []

    documents = []
    for item in data["policies"]:
        title = item.get("title")
        gdrive_id = item.get("gdrive_id")
        filename = os.path.join(DOCUMENTS_DIR, f"{title}.pdf")

        # Download skipped for now (assuming files are present)

        try:
            with pdfplumber.open(filename) as pdf:
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