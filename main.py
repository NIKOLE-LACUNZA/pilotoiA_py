import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from loader import cargar_pdf
from vector_store import crear_vectorstore, responder_pregunta
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Carga el PDF y crea el vectorstore al iniciar
texto = cargar_pdf("documentos/DS009-2025-EF-Reglamento-ley-de-contrataciones.publicas.pdf")
vectorstore = crear_vectorstore(texto)

class Pregunta(BaseModel):
    mensaje: str

@app.post("/api/chat")
def chat(pregunta: Pregunta):
    try:
        respuesta = responder_pregunta(pregunta.mensaje, vectorstore)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": str(e)}