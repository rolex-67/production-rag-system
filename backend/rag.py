import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

load_dotenv()

# Always resolve the chroma_db path correctly:
# - Inside Docker: /app/chroma_db  (set via environment variable CHROMA_DB_PATH)
# - Running locally: ./chroma_db relative to the project root (one level up from backend/)
DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "chroma_db")
)

class ProductionRAG:
    def __init__(self, persist_directory=DEFAULT_CHROMA_PATH):
        persist_directory = os.path.abspath(persist_directory)
        print(f"Using ChromaDB at: {persist_directory}")
        
        # Use REST transport to avoid gRPC timeout issues (especially in Docker on Windows)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            transport="rest"
        )
        
        # 1. Load Vector Store
        print("Loading ChromaDB...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. Load BM25 Retriever for Hybrid Search
        print("Setting up BM25 Retriever for Hybrid Search...")
        db_data = self.vectorstore.get()
        if not db_data['documents']:
            print("WARNING: ChromaDB is empty. Please run indexer.py first.")
            self.docs = []
            self.bm25_retriever = None
        else:
            self.docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(db_data['documents'], db_data['metadatas'])
            ]
            self.bm25_retriever = BM25Retriever.from_documents(self.docs)
            self.bm25_retriever.k = 3
            print(f"BM25 initialized with {len(self.docs)} documents.")
            
        # 3. Create Ensemble Retriever (Hybrid Search: Vector + BM25)
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
        else:
            self.ensemble_retriever = self.vector_retriever
        
        # 4. Initialize LLM — REST transport avoids Docker gRPC network timeouts
        # max_retries=0 means quota/rate-limit errors surface immediately (no 5-min hang)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0,
            transport="rest",
            max_retries=0
        )
        
        # 5. Define RAG Prompt — explicitly tells model to say "I don't know"
        self.prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant answering questions based on the provided context.
            
Context: {context}

Question: {question}

Instructions:
- Answer the question based strictly on the context above.
- If the context does not contain the answer, say: "I don't know based on the provided context." Do NOT hallucinate.
- Keep your answer concise and accurate.

Answer:"""
        )
        
        # Build the LLM Chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Guardrails LLM — same model, separate instance for clarity
        self.guardrail_llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0,
            transport="rest",
            max_retries=0
        )

    def check_guardrails(self, query: str) -> bool:
        """
        Guardrail to prevent PII or off-topic queries.
        Returns True if the query is SAFE, False if it should be BLOCKED.
        """
        guardrail_prompt = PromptTemplate.from_template(
            """Analyze the following user query. Respond with ONLY 'SAFE' or 'BLOCKED'.
Block the query if it contains sensitive PII (SSN, credit cards, passwords) or is malicious/inappropriate.

Query: {query}

Response:"""
        )
        chain = guardrail_prompt | self.guardrail_llm | StrOutputParser()
        result = chain.invoke({"query": query})
        return "SAFE" in result.upper()

    def query(self, question: str) -> dict:
        # Step 1: Guardrails Check
        if not self.check_guardrails(question):
            return {
                "answer": "⚠️ Query blocked by guardrails. Please avoid sensitive PII or inappropriate requests.",
                "context": []
            }
            
        # Step 2: Hybrid Retrieval (Vector + BM25)
        docs = self.ensemble_retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]
        context_string = "\n\n".join(context_texts)
        
        # Step 3: Generate Answer via LLM
        answer = self.chain.invoke({
            "context": context_string,
            "question": question
        })
        
        return {
            "answer": answer,
            "context": context_texts
        }
