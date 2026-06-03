import argparse
import os
from functools import partial
from lightrag import LightRAG
from utils import extract_text_from_pdf
from lightrag import QueryParam



arg_parser = argparse.ArgumentParser(description="Run LightRAG worker")
arg_parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "ollama"),
                        help="LLM provider: 'ollama', 'openai', or 'gemini'")

subparsers = arg_parser.add_subparsers(dest="command")
index_parser = subparsers.add_parser("index", help="Index a PDF document")
index_parser.add_argument("pdf_path", type=str, help="Path to the PDF document to index")

query_parser = subparsers.add_parser("query", help="Query the indexed documents")
query_parser.add_argument("question", type=str, help="Text of the query")
query_parser.add_argument("--mode", type=str, default="hybrid", help="Query mode: 'hybrid', 'embedding', or 'llm'")

def _get_llm_model_funcs(provider):
        if provider == "openai":
            from lightrag.llm.openai import openai_complete, openai_embed
            return openai_complete, openai_embed
        elif provider == "gemini":
            from lightrag.llm.gemini import gemini_model_complete, gemini_embed
            from lightrag.utils import EmbeddingFunc

            embed_func = EmbeddingFunc(
                embedding_dim=int(os.getenv("GEMINI_EMBEDDING_DIM", "3072")),
                max_token_size=2048,
                func=gemini_embed.func,
            )
            return gemini_model_complete, embed_func
        elif provider == "ollama":
            from lightrag.llm.ollama import ollama_embed, ollama_model_complete
            from lightrag.utils import EmbeddingFunc

            model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            embed_func = EmbeddingFunc(
                embedding_dim=int(os.getenv("OLLAMA_EMBEDDING_DIM")) if os.getenv("OLLAMA_EMBEDDING_DIM") else (768 if "nomic" in model_name else 1024),
                max_token_size=8192,
                func=partial(
                ollama_embed.func,
                embed_model=model_name,
                host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            )
            return ollama_model_complete, embed_func
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            

if __name__ == "__main__":
    args = arg_parser.parse_args()
    working_dir = f"./lightrag_store_{args.provider}"
    llm_func, embed_func = _get_llm_model_funcs(args.provider)
    rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func, # extract entities and relations
            llm_model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if args.provider == "gemini" else os.getenv("OLLAMA_MODEL", "llama3"),
            embedding_func=embed_func, # create embeddings for query and documents
            default_llm_timeout=600,
        )
    import asyncio
    asyncio.run(rag.initialize_storages())

    if args.command == "index":
        print(f"Indexing document: {args.pdf_path}")
        text = extract_text_from_pdf(args.pdf_path)
        rag.insert(text)
        print("Indexing complete.")
    elif args.command == "query":
        import sys
        print(f"Querying with question: {args.question}", file=sys.stderr)
        answer = rag.query(args.question, param=QueryParam(mode=args.mode))
        print(answer or "")
    else:
        print("No command specified. Use 'index' or 'query'.")