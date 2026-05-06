import asyncio
import os
import pickle
from pathlib import Path
from typing import Optional

import nest_asyncio
from functools import partial
nest_asyncio.apply()

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from llm import ask_llm
import warnings
warnings.filterwarnings("ignore")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INDEX_DIR = "vector_store"
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


class RAGAdapter(ABC):
    # @abstractmethod
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() or ""
        return pdf_text
    
    @abstractmethod
    def index_document(self, pdf_path: str) -> None:
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        pass

class LightRAGAdapter(RAGAdapter):
    def __init__(self, working_dir="./lightrag_store"):
        from lightrag import LightRAG

        llm_func, embed_func = self._get_llm_model_funcs(os.getenv("LLM_PROVIDER", "ollama"))
        self.graphRAG = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            llm_model_name=os.getenv("OLLAMA_MODEL", "llama3"),
            embedding_func=embed_func,
            default_llm_timeout=600,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.graphRAG.initialize_storages())

    def _run_async(self, coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    def index_document(self, pdf_path: str) -> None:
        text = self.extract_text_from_pdf(pdf_path)
        self._run_async(self.graphRAG.ainsert(text))

    def query(self, query: str, mode='hybrid') -> str:
        from lightrag import QueryParam
        result = self.graphRAG.query(query, param=QueryParam(mode=mode))
        return result or "No relevant information found."

    def _get_llm_model_funcs(self, provider):
        if provider == "openai":
            from lightrag.llm.openai import openai_complete, openai_embedding
            return openai_complete, openai_embedding
        elif provider == "gemini":
            from lightrag.llm.gemini import gemini_complete, gemini_embedding
            return gemini_complete, gemini_embedding
        elif provider == "ollama":
            from lightrag.llm.ollama import ollama_model_complete, ollama_embed
            from lightrag.utils import EmbeddingFunc

            model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            embed_func = EmbeddingFunc(
                embedding_dim=768 if "nomic" in model_name else 1024,
                max_token_size=8192,
                func=partial(
                ollama_embed.func,  # Access the unwrapped function to avoid double EmbeddingFunc wrapping
                embed_model=model_name,
                host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
                # model_name=model_name,
            )
            return ollama_model_complete, embed_func
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class BasicRAGAdapter(RAGAdapter):
    model = SentenceTransformer(EMB_MODEL_NAME)
    def __init__(self):
        self.index = None
        self.all_chunks = []
        self.top_k = 3
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def index_document(self, pdf_path: str) -> None:
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        self.all_chunks.extend(chunks)
        self.build_vector_store()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def build_vector_store(self):
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

    def load_vector_store(self):
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return None
        
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.all_chunks = pickle.load(f)

    def query(self, query: str) -> str:
        if self.index is None:
            return "No documents indexed yet. Please upload and index a PDF first."
        query_embedding = self.model.encode([query]).astype("float32")
        _distances, indices = self.index.search(query_embedding, self.top_k)
        return ask_llm(query, [self.all_chunks[i] for i in indices[0] if i < len(self.all_chunks)])
