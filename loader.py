from PyPDF2 import PdfReader

def cargar_pdf(ruta_pdf: str) -> str:
    texto = " "
    lector = PdfReader(ruta_pdf)
    for pagina in lector.pages:
        texto += pagina.extract_text() or ""
    return texto