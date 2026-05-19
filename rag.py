import os
import pickle
import subprocess

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from classify_questions import classify_questions
from abc import ABC, abstractmethod
from llm import ask_llm
from utils import extract_text_from_pdf
import warnings
warnings.filterwarnings("ignore")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INDEX_DIR = "vector_store"
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


class RAGAdapter(ABC):
    @abstractmethod
    def index_document(self, pdf_path: str) -> None:
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        pass

class AutoLightRAGAdapter(RAGAdapter):
    def __init__(self):
        self.light_adapter = LightRAGAdapter()

    def index_document(self, pdf_path: str) -> None:
        self.light_adapter.index_document(pdf_path)

    def query(self, question: str) -> str:
        category = classify_questions(question)
        if category == "summary":
            return self.light_adapter.query(question, mode="global")
        # if category == "relation":
        #     return self.light_adapter.query(question, mode="hybrid")
        else:
            return self.light_adapter.query(question, mode = "hybrid")

class LightRAGAdapter(RAGAdapter):
    def index_document(self, pdf_path):
          subprocess.run(["uv", "run", "python", "lightrag_worker.py", "index", pdf_path])

    def query(self, question, mode="hybrid"):
          result = subprocess.run(
              ["uv", "run", "python", "lightrag_worker.py", "query", question, "--mode", mode],
              capture_output=True, text=True
          )
          return result.stdout.strip() or "No relevant information found."

class BasicRAGAdapter(RAGAdapter):
    model = SentenceTransformer(EMB_MODEL_NAME)
    def __init__(self):
        self.index = None
        self.all_chunks = []
        self.top_k = 3
    
    def index_document(self, pdf_path: str) -> None:
        text = extract_text_from_pdf(pdf_path)
        chunks = self._chunk_text(text)
        self.all_chunks.extend(chunks)
        self._build_vector_store()

    def query(self, query: str) -> str:
        if self.index is None:
            return "No documents indexed yet. Please upload and index a PDF first."
        query_embedding = self.model.encode([query]).astype("float32")
        _distances, indices = self.index.search(query_embedding, self.top_k)
        return ask_llm(query, [self.all_chunks[i] for i in indices[0] if i < len(self.all_chunks)])

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_vector_store(self):
        embeddings = self.model.encode(self.all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # Save to disk
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump(self.all_chunks, f)
        print(f"Indexed {len(self.all_chunks)} chunks. Vector store saved to {INDEX_DIR}.")

    def _load_vector_store(self):
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return None
        
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.all_chunks = pickle.load(f)

    
