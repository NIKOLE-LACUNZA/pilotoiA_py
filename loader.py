from PyPDF2 import PdfReader


def cargar_pdf(path: str) -> str:
    with open(path, "rb") as file:
        reader = PdfReader(file)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() or ""
        return texto

def cargar_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() or ""
    return texto