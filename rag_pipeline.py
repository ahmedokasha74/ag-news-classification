"""Optional Groq and RAG helpers for richer news explanations."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from dotenv import load_dotenv

from data_loader import LABEL_MAP
from ml_model import predict_category


load_dotenv()

def get_groq_client(api_key: Optional[str] = None):
    """Create a Groq client from an explicit key or GROQ_API_KEY."""
    from groq import Groq

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Set GROQ_API_KEY before calling Groq analysis.")
    return Groq(api_key=key)


def build_vector_store(documents: Iterable[str], model_name: str = "all-MiniLM-L6-v2"):
    """Build a FAISS vector store for retrieval-augmented generation."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(list(documents))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(docs, embeddings)


class NewsAnalyzer:
    """Classify news and optionally ask an LLM to explain the result."""

    def __init__(self, classifier, vector_store=None, groq_api_key: Optional[str] = None):
        self.classifier = classifier
        self.vector_store = vector_store
        self.groq_api_key = groq_api_key

    def classify(self, text: str) -> tuple[int, str]:
        """Return numeric and human-readable category labels."""
        return predict_category(self.classifier, text)

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Return similar training snippets when a vector store is available."""
        if self.vector_store is None:
            return ""
        results = self.vector_store.similarity_search(query, k=k)
        return "\n".join(result.page_content for result in results)

    def analyze_with_groq(self, text: str, model: str = "llama-3.1-8b-instant") -> str:
        """Generate a plain-English analysis with Groq."""
        label_id, label_name = self.classify(text)
        context = self.retrieve_context(f"{label_name} news: {text}")
        prompt = f"""
News category: {label_name} ({label_id})

Related examples:
{context or "No retrieved examples were provided."}

News text:
{text}

Explain why the item belongs to this category, summarize it, list key entities,
and describe the likely sentiment.
"""
        client = get_groq_client(self.groq_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


def label_name(label: int) -> str:
    """Small convenience wrapper for labels."""
    return LABEL_MAP[int(label)]
