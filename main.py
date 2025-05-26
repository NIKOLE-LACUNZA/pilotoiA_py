import os
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from vector_store import responder_pregunta
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()  # Solo tiene efecto localmente

app = FastAPI()

# Configurar CORS
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
    print("Clave OpenAI cargada:", "✅" if openai.api_key else "❌")

    # Solo carga el vectorstore ya guardado (rápido)
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
