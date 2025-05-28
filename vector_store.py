import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI




def crear_vectorstore(texto: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(fragmentos, embedding=embeddings)
    return vectorstore
def crear_y_guardar_vectorstore(texto: str, carpeta: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(fragmentos, embedding=embeddings)
    vectorstore.save_local(carpeta)
    
def procesar_y_guardar_vectorstore(contenido_pdf: bytes, ruta_vectorstore: str):
    import io
    from loader import cargar_pdf_bytes
    from vector_store import crear_y_guardar_vectorstore
    texto = cargar_pdf_bytes(contenido_pdf)
    crear_y_guardar_vectorstore(texto, ruta_vectorstore)

def responder_pregunta(pregunta: str, vectorstore: FAISS) -> str:
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return chain.run(pregunta)
