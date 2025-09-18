import io, os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import docx2txt

def read_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def read_pdf_text(path: str) -> str:
    """Extract selectable text; fallback is image OCR per-page elsewhere."""
    with fitz.open(path) as doc:
        return "\n".join([page.get_text("text") for page in doc])

def pdf_to_images(path: str, dpi: int = 300):
    return convert_from_path(path, dpi=dpi)

def image_bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")
