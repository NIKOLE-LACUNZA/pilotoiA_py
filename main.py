import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from loader import cargar_pdf
from vector_store import crear_vectorstore, responder_pregunta
from dotenv import load_dotenv
import os
from vector_store import crear_y_guardar_vectorstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()  # En local, permite cargar variables del archivo .env

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura la API Key al iniciar la aplicación
@app.on_event("startup")
def startup_event():
    global vectorstore
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("Clave OpenAI cargada:", "✅" if openai.api_key else "❌ (no encontrada)")

    # Carga y vectoriza el documento
    
texto = cargar_pdf("documentos/DS009-2025-EF-Reglamento-ley-de-contrataciones.publicas.pdf")
crear_y_guardar_vectorstore(texto, "vector_db")

vectorstore = FAISS.load_local(
        "vector_db",
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True
    )
class Pregunta(BaseModel):
    mensaje: str

@app.post("/api/chat")
def chat(pregunta: Pregunta):
    try:
        respuesta = responder_pregunta(pregunta.mensaje, vectorstore)
        return {"respuesta": respuesta}
    except Exception as e:
        return {"error": str(e)}
