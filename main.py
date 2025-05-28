import os
import io
import openai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
from loader import cargar_pdf, cargar_pdf_bytes
from loader import cargar_pdf_bytes
from vector_store import crear_y_guardar_vectorstore, responder_pregunta
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from fastapi import BackgroundTasks
from vector_store import procesar_y_guardar_vectorstore

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
    documento: str

# Endpoint principal
@app.post("/api/chat")
def chat(pregunta: Pregunta):
    try:
        ruta_vectorstore = f"./vectorstores/vector_db_{pregunta.documento}" 
        if not os.path.exists(os.path.join(ruta_vectorstore, "index.faiss")):
            return {
                "error": f"El vectorstore para '{pregunta.documento}' aún no está listo. Intenta en unos segundos."
            }
        vectorstore = FAISS.load_local(
            ruta_vectorstore,
            OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
            allow_dangerous_deserialization=True
        )
        respuesta = responder_pregunta(pregunta.mensaje, vectorstore)
        return {"respuesta": respuesta}
    except Exception as e:
        import traceback
        print("❌ Excepción en /api/chat:", traceback.format_exc())
        return {"error": str(e)}
    
@app.post("/api/subir-documento")
async def subir_documento(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    try:
        contenido_pdf = await file.read()
        nombre_base = os.path.splitext(file.filename)[0].replace(" ", "_")
        ruta_vectorstore = f"./vectorstores/vector_db_{nombre_base}"
        os.makedirs("vectorstores", exist_ok=True) 
        # Lanza la tarea en background (sin bloquear la respuesta)
        background_tasks.add_task(procesar_y_guardar_vectorstore, contenido_pdf, ruta_vectorstore)

        # Responde rápido sin esperar que termine el proceso pesado
        return JSONResponse(
            content={"mensaje": f"Vectorstore se está creando en background para {ruta_vectorstore}"},
            status_code=202
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

