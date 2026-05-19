import os
from llm import LLM_PROVIDER, get_model_name, get_llm_client


def classify_questions(question: str) -> str:
    """
    Classify questions into categories: 'summary', 'relation', 'specific'.

    Args:
        question (str): A single question string.

    Returns:
        str: The classification of the question.
    """
    # Placeholder implementation - replace with actual classification logic
    summary_keywords = ["summary", "summarize", "whole", "overall"]
    relation_keywords = ["relate", "relationship", "connect", "why", "how", "related"]
    specific_keywords = [
        "specific",
        "detail",
        "details",
        "particular",
        "which",
        "who",
        "what",
        "when",
        "where",
        "how many",
    ]
    if any(keyword in question.lower() for keyword in summary_keywords):
        category = "summary"
    elif any(keyword in question.lower() for keyword in relation_keywords):
        category = "relation"
    elif any(keyword in question.lower() for keyword in specific_keywords):
        category = "specific"
    else:
        client = get_llm_client()
        model_name = get_model_name()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"""Classify this question into one category. Reply with ONLY the category name.

                            Categories:
                            - specific: asks about exact text, quotes, numbers, dates
                            - relationship: asks how things connect or relate
                            - global: asks for summary or overview

                        Question: {question}""",
                }
            ],
            temperature=0.0,
        )
        category = response.choices[0].message.content.strip()

    return category
