from PyPDF2 import PdfReader
import io

def cargar_pdf(path: str) -> str:
    with open(path, "rb") as file:
        reader = PdfReader(file)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() or ""
        return texto

def cargar_pdf_bytes(pdf_bytes: bytes) -> str:
    stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(stream)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() or ""
    return texto