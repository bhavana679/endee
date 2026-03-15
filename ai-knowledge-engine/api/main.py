import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the project root to the system path to allow importing internal modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag.rag_pipeline import RAGPipeline
from embeddings.embed_store import EmbeddingPipeline
from retrieval.endee_client import EndeeClient
from config.config import ENDEE_HOST, ENDEE_PORT

app = FastAPI(
    title="AI Knowledge Engine API",
    description="Backend API for Retrieval Augmented Generation using Endee Vector DB",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Welcome to the AI Engineering Knowledge Memory Engine API!",
        "docs": "/docs",
        "health": "/health"
    }

# Initialize application dependencies
rag_pipeline = RAGPipeline()
endee_client = EndeeClient(host=ENDEE_HOST, port=int(ENDEE_PORT))

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/health")
def health_check():
    """Confirms API connectivity and checks the local Endee database health."""
    try:
        endee_status = endee_client.health()
        return {
            "status": "healthy",
            "api": "online",
            "endee_connected": True,
            "endee_status": endee_status
        }
    except Exception as e:
        return {
            "status": "degraded",
            "api": "online",
            "endee_connected": False,
            "error": str(e)
        }

@app.post("/ingest")
def trigger_ingestion():
    """Triggers the Embedding Ingestion pipeline for all document files."""
    try:
        pipeline = EmbeddingPipeline()
        pipeline.run_ingestion()
        return {"status": "success", "message": "Ingestion pipeline completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")

@app.post("/query")
def query_knowledge_base(request: QueryRequest):
    try:
        result = rag_pipeline.answer_question(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
