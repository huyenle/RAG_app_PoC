# RAG App PoC

A PDF chatbot that uses Retrieval-Augmented Generation to answer questions from uploaded documents.

## Features

- **Two RAG backends** switchable from the UI:
  - **Basic RAG** -- FAISS vector search + sentence-transformers embeddings
  - **Graph RAG** -- LightRAG knowledge graph with entity/relationship extraction
- **Multiple LLM providers** -- Ollama (local/free), OpenAI, Anthropic, Google Gemini
- **Streamlit UI** -- upload PDFs, select documents, chat interface
- **Per-document adapters** -- each PDF can use a different RAG method
- **Reprocess documents** -- switch RAG method on already-uploaded files

## Setup

```bash
uv sync
cp .env.example .env   # edit with your settings
```

### LLM setup (pick one)

**Ollama (free, local):**
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

**Paid API:** Set `LLM_PROVIDER` and API key in `.env`.

## Run

```bash
make run
```

Opens at http://localhost:8501

## Project structure

```
app.py    -- Streamlit UI
rag.py    -- RAGAdapter (ABC), BasicRAGAdapter, LightRAGAdapter
llm.py    -- LLM provider configuration
```
