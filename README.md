# 🚀 Production-Ready RAG System

A domain-specific, production-ready Retrieval-Augmented Generation (RAG) system with a FastAPI backend and a Streamlit frontend. It allows users to upload any PDF document, automatically indexes it using semantic chunking, and provides an interactive chat interface to ask questions about the document.

![UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Backend API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## ✨ Key Features
- **Dynamic Document Upload:** Drag-and-drop PDF upload directly from the UI.
- **Semantic Chunking:** Splits text based on semantic meaning rather than arbitrary character counts, ensuring context is preserved.
- **Hybrid Search:** Combines Vector Search (ChromaDB) and Keyword Search (BM25) via an Ensemble Retriever for highly accurate results.
- **Guardrails:** Prevents PII (Personally Identifiable Information) leaks and off-topic queries.
- **Confidence Thresholds:** The LLM is explicitly prompted to say "I don't know" when the answer isn't in the context, preventing hallucinations.
- **Observability:** Integrated with LangSmith for end-to-end tracing and evaluation.

## 🛠️ Technology Stack & Architecture

| Component | Technology | Why we use it |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) | Enables rapid UI development for AI/Data applications with built-in file uploaders and interactive widgets. |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async web framework for Python. Built-in Pydantic validation makes it robust for production APIs. |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Industry-standard framework for chaining LLM calls, retrievers, and chunking strategies. |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Open-source, lightweight, and local vector database. Perfect for fast semantic search without external dependencies. |
| **LLM & Embeddings** | [Google Gemini](https://ai.google.dev/) | Provides state-of-the-art context windows, fast inference (Flash models), and high-quality embeddings. |
| **Evaluation/Tracing** | [LangSmith](https://smith.langchain.com/) | Critical for productionizing LLMs. Allows us to trace exact token usage, latency, and evaluate answer quality. |
| **Infrastructure** | [Docker & Compose](https://www.docker.com/) | Ensures the system works perfectly across any operating system (Windows, Mac, Linux) by containerizing dependencies. |

## ⚙️ Local Setup & Installation

### 1. Prerequisites
- Docker & Docker Compose installed
- Python 3.10+ (if running natively)
- A Google Gemini API Key

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT="production-rag"
```

### 3. Run with Docker (Recommended)
This will spin up both the FastAPI backend and Streamlit frontend in isolated containers.
```bash
docker-compose up --build
```
- UI is available at: `http://localhost:8501`
- API is available at: `http://localhost:8000`

## 🧠 System Workflow

1. **Ingestion**: User uploads a PDF. `PyPDFLoader` extracts the text.
2. **Chunking**: `SemanticChunker` groups sentences by semantic similarity, avoiding cutting paragraphs mid-thought.
3. **Storage**: The chunks are embedded using `gemini-embedding-001` and stored in ChromaDB locally.
4. **Querying**: User asks a question. The query is routed through a Guardrails LLM to check for PII/malicious intent.
5. **Retrieval**: The `EnsembleRetriever` fetches the top chunks using both dense vector similarity and sparse BM25 keyword matching.
6. **Generation**: The retrieved context and question are sent to `gemini-3.8-flash` with strict anti-hallucination prompts to generate the final answer.
