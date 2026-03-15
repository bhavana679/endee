# AI Engineering Knowledge Memory Engine

A high-performance, Retrieval-Augmented Generation (RAG) platform designed as a "second brain" for AI Engineers. Built on the **Endee Vector Database**, this engine allows users to index specialized technical documentation and retrieve contextually accurate answers with full traceability.

---

## Architecture

The system follows a modular, scalable pipeline to ensure rapid retrieval and high-quality generation:

1.  **Ingestion:** Raw documents (`.txt`) are processed, cleaned, and split into semantic chunks.
2.  **Embedding:** Chunks are transformed into 384-dimensional vectors using the `all-MiniLM-L6-v2` SentenceTransformer.
3.  **Storage:** Vectors and metadata are stored in the **Endee Vector Database** with cosine similarity metrics.
4.  **Retrieval:** User queries are embedded and matched against the database to find the top K most relevant contexts.
5.  **RAG Pipeline:** Contextual snippets are injected into a specialized prompt and sent to an LLM (GPT-4o/mini) for answer generation.
6.  **Backend API:** A **FastAPI** layer facilitates communication between the database, the AI model, and the frontend.
7.  **Frontend UI:** A clean **Streamlit** interface provides a chat-based experience for the end-user.

---

##  Features

*   **Semantic Search:** Deep architectural search beyond simple keywords.
*   **Vector Embeddings:** State-of-the-art dense vector representation.
*   **Retrieval Augmented Generation:** Answers generated strictly based on provided technical documentation.
*   **Inline Source Citations:** Every answer includes bracketed references (e.g., `[1]`, `[2]`) linked to source files.
*   **Retrieval Diagnostics:** Real-time visibility into similarity distances and ranking metadata.
*   **Query Observability:** End-to-end tracking of response times and source utilization.
*   **FastAPI Backend:** Production-ready RESTful API.
*   **Knowledge Manager:** Trigger re-indexing of the entire knowledge base directly from the UI.

---

##  Technology Stack

*   **Language:** Python 3.12+
*   **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
*   **Vector Database:** [Endee](https://github.com/endeeio/endee)
*   **LLM Integration:** OpenAI API
*   **Backend:** FastAPI & Pydantic
*   **Frontend:** Streamlit

---

##  Setup Instructions

### 1. Start the Endee Vector Database
Ensure you have Docker installed and run:
```bash
docker run -d --ulimit nofile=100000:100000 -p 8080:8080 -v ./data:/data --name endee-server endeeio/endee-server:latest
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Platform

**Step A: Ingest Knowledge** (One-time or when data changes)
```bash
python embeddings/embed_store.py
```

**Step B: Launch Backend API**
```bash
python3 -m uvicorn api.main:app --reload --port 8000
```

**Step C: Launch Streamlit UI**
```bash
streamlit run ui/app.py
```

---

## Folder Structure

```text
ai-knowledge-engine/
├── api/                # FastAPI application & routes
├── config/             # Environment & model configurations
├── data/               # Raw engineering documents (.txt)
├── embeddings/         # Document chunking & vector ingestion logic
├── rag/                # RAG orchestration & prompt engineering
├── retrieval/          # Endee client & semantic search logic
├── ui/                 # Streamlit interface
├── utils/              # Text processing & cleaning utilities
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

##  Example Queries

*   "What is the modular architecture of the RAG engine?"
*   "How are text chunks handled in the embedding pipeline?"
*   "What was the fix for the vector precision bug?"

---

##  Future Improvements

*   **Multi-modal Support:** Indexing PDFs, Markdown, and technical diagrams.
*   **Hybrid Search:** Combining semantic search with BM25 keyword matching for better precision.
*   **Local LLM Support:** Integration with Ollama or vLLM for fully air-gapped operations.
*   **User Authentication:** Multi-tenant support for private knowledge bases.

---

