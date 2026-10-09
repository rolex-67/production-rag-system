import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Default chroma_db path — one level up from this file (project root)
DEFAULT_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def index_pdf(pdf_path: str, persist_directory: str = DEFAULT_CHROMA_PATH) -> int:
    """
    Load a PDF, split it into semantic chunks, embed them, and store in ChromaDB.
    Returns the number of chunks created.
    """
    persist_directory = os.path.abspath(persist_directory)
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Initializing Semantic Chunker...")
    # REST transport — avoids gRPC issues in Docker
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        transport="rest"
    )
    text_splitter = SemanticChunker(embeddings)

    print("Splitting document into semantic chunks...")
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"Storing chunks in ChromaDB at: {persist_directory}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Indexing complete!")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a PDF document using Semantic Chunking")
    parser.add_argument("pdf_path", help="Path to the PDF file to index")
    args = parser.parse_args()
    index_pdf(args.pdf_path)
