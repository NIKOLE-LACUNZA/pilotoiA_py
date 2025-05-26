import os
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from vector_store import responder_pregunta
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno (solo en local)
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global
vectorstore = None

@app.on_event("startup")
def startup_event():
    global vectorstore
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("🔑 Clave OpenAI cargada:", "✅" if openai.api_key else "❌ (no encontrada)")

    try:
        vectorstore = FAISS.load_local(
            "vector_db",
            OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            allow_dangerous_deserialization=True
        )
        print("✅ Vectorstore cargado correctamente.")
    except Exception as e:
        print("❌ Error al cargar vectorstore:", e)

# Ruta raíz para pruebas simples desde navegador o Azure
@app.get("/")
def home():
    return {"status": "ok", "message": "API PilotoIA funcionando correctamente"}

# Modelo de entrada
class Pregunta(BaseModel):
    mensaje: str

# Endpoint principal
@app.post("/api/chat")
def chat(pregunta: Pregunta):
    try:
        if not vectorstore:
            return {"error": "Vectorstore no está cargado"}
        respuesta = responder_pregunta(pregunta.mensaje, vectorstore)
        return {"respuesta": respuesta}
    except Exception as e:
        import traceback
        print("❌ Excepción en /api/chat:", traceback.format_exc())
        return {"error": str(e)}
