import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


def get_llm_client() -> OpenAI:
    if LLM_PROVIDER == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif LLM_PROVIDER == "anthropic":
        return OpenAI(
            base_url="https://api.anthropic.com/v1/",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif LLM_PROVIDER == "gemini":
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
            api_key="ollama",
        )


def get_model_name() -> str:
    if LLM_PROVIDER == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif LLM_PROVIDER == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    elif LLM_PROVIDER == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return os.getenv("OLLAMA_MODEL", "llama3")


def ask_llm(question: str, context_chunks: list[str]) -> str:
    client = get_llm_client()
    context = "\n\n---\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided document context. "
                    "Only use the information from the context below to answer. "
                    "If the answer is not in the context, say you don't have enough information.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
