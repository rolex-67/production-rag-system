import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from rag import ProductionRAG
from indexer import index_pdf
import uvicorn

app = FastAPI(title="Production RAG API")

# Global RAG system — re-created after each upload
rag_system = None

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/data")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


def init_rag():
    """Helper to (re)initialize the RAG system."""
    global rag_system
    rag_system = ProductionRAG(persist_directory=os.path.abspath(CHROMA_DB_PATH))


@app.on_event("startup")
def startup_event():
    init_rag()


# ─── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: List[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
    num_chunks: int


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file, index it into ChromaDB, and reload the RAG system.
    The user can start querying the file immediately after this returns.
    """
    # Only accept PDF files
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Index the PDF (this creates/updates ChromaDB)
    try:
        num_chunks = index_pdf(save_path, persist_directory=os.path.abspath(CHROMA_DB_PATH))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    # Reload the RAG system so it sees the new documents
    init_rag()

    return UploadResponse(
        message=f"✅ '{file.filename}' uploaded and indexed successfully!",
        filename=file.filename,
        num_chunks=num_chunks
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized.")
    try:
        result = rag_system.query(req.query)
        return QueryResponse(answer=result["answer"], context=result["context"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
