"""Text cleaning helpers for the AG News project."""

from __future__ import annotations

import re


def clean_text(text: object) -> str:
    """Return a normalized version of a news headline/body string."""
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

