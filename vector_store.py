from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def crear_vectorstore(texto: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_text(texto)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(fragmentos, embedding=embeddings)
    return vectorstore

def responder_pregunta(pregunta: str, vectorstore: FAISS) -> str:
    llm = ChatOpenAI(model_name="gpt-4")
    chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    return chain.run(pregunta)