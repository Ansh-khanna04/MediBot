import os
import shutil
from fastapi import FastAPI , UploadFile ,File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from config import VECTOR_STORE_PATH
from ingest import ingest
from chat import load_chain
from pathlib import Path


# APP SETUP
chain = None
retriever = None
chat_history = []
UPLOAD_DIR = Path(__file__).resolve().parent / "uploaded_pdfs"
SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# Lifespan : runs on startup
@asynccontextmanager
async def lifespan(app:FastAPI):
    global chain, retriever
    index_path = Path(VECTOR_STORE_PATH) / "index.faiss"
    if index_path.exists():
        print("Vector store found - loading chain")
        chain,retriever = load_chain()
        print("MediBOT is ready")
    else:
        print("No Vector store found - ingesting sample data...")
        if SAMPLE_DATA_DIR.exists():
            ingest(str(SAMPLE_DATA_DIR))
            chain,retriever = load_chain()
            print("SAMple data ingested and chain loaded")
        else:
            print(" No sample data found either. Upload PDFs via /upload.")
    yield # App runs here

# APP
app = FastAPI(title = "MediBOT API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- ENDpoint 1 - Health Check----

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store_loaded":chain is not None
    }

# ---Endpoint 2: Upload PDfs---
@app.post("/upload")
async def upload_pdfs(files:list[UploadFile] = File(...)):
    global  chain,retriever,chat_history

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail = f"{file.filename} is not a PDF"
            )
        dest = UPLOAD_DIR / file.filename
        with open(dest,"wb") as f:
            shutil.copyfileobj(file.file,f)
        saved.append(file.filename)
        print(f"saved: {file.filename}")

    print(f"Ingesting{len(saved)} uploaded files")
    ingest(str(UPLOAD_DIR))
    chain, retriever= load_chain()
    chat_history =[]

    return {
        "message":f"Successfully ingested {len(saved)} uploaded files"
    }

# ----Endpoint 3 : Chat---

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: QuestionRequest):
    global chain, retriever, chat_history

    if chain is None:
        raise HTTPException(status_code=400, detail="No documents loaded yet.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Get source docs for citations
    source_docs = retriever.invoke(req.question)

    # Get answer
    answer = chain.invoke(req.question)

    # Update chat history
    chat_history.append(HumanMessage(content=req.question))
    chat_history.append(AIMessage(content=answer))

    # Build citations
    seen = set()
    sources = []
    for doc in source_docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            })

    return {"answer": answer, "sources": sources}

