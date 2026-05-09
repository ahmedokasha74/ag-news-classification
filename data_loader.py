"""Dataset loading utilities for AG News classification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset

from text_preprocessing import clean_text


LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}


def _read_ag_news_csv(path: Path) -> pd.DataFrame:
    """Read the original AG News CSV format and return standard columns."""
    df = pd.read_csv(path, header=None)
    df.columns = ["label", "title", "description"]
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    return df[["label", "text"]]


def _load_from_local_csv(data_dir: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load train/test CSV files when they already exist locally."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return _read_ag_news_csv(train_path), _read_ag_news_csv(test_path)
    return None


def _load_from_huggingface() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load AG News from Hugging Face as a convenient fallback."""
    dataset = load_dataset("ag_news")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    train_df["label"] = train_df["label"] + 1
    test_df["label"] = test_df["label"] + 1
    return train_df[["label", "text"]], test_df[["label", "text"]]


def load_ag_news(
    data_dir: str | Path = ".",
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    clean: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load AG News from local CSV files or Hugging Face.

    Args:
        data_dir: Folder containing optional ``train.csv`` and ``test.csv`` files.
        train_limit: Optional number of train rows for quick demos.
        test_limit: Optional number of test rows for quick demos.
        clean: Whether to add cleaned text.
    """
    data_dir = Path(data_dir)
    loaded = _load_from_local_csv(data_dir)
    train_df, test_df = loaded if loaded else _load_from_huggingface()

    if train_limit:
        train_df = train_df.sample(train_limit, random_state=42)
    if test_limit:
        test_df = test_df.sample(test_limit, random_state=42)

    train_df = train_df.copy()
    test_df = test_df.copy()
    if clean:
        train_df["clean_text"] = train_df["text"].apply(clean_text)
        test_df["clean_text"] = test_df["text"].apply(clean_text)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

