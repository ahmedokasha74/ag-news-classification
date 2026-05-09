"""Colab-ready Gradio demo for AG News classification."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from data_loader import load_ag_news
from ml_model import evaluate_ml_model, load_model, predict_category, save_model, train_ml_model
from rag_pipeline import NewsAnalyzer


load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "ml_model.joblib"))
TRAIN_LIMIT = int(os.getenv("TRAIN_LIMIT", "12000"))
TEST_LIMIT = int(os.getenv("TEST_LIMIT", "2000"))


def get_or_train_model():
    """Load a saved model or train a quick demo model."""
    if MODEL_PATH.exists():
        return load_model(str(MODEL_PATH)), "Loaded saved model."

    train_df, test_df = load_ag_news(train_limit=TRAIN_LIMIT, test_limit=TEST_LIMIT)
    model = train_ml_model(train_df)
    metrics = evaluate_ml_model(model, test_df)
    save_model(model, str(MODEL_PATH))
    return model, f"Trained demo model. Accuracy: {metrics['accuracy']:.3f}"


MODEL, STATUS = get_or_train_model()
ANALYZER = NewsAnalyzer(MODEL)


def classify_news(text: str, include_llm_analysis: bool = False):
    """Classify one news item and optionally run Groq explanation."""
    if not text.strip():
        return "Please enter a news headline or paragraph.", ""

    _, category = predict_category(MODEL, text)
    prediction = f"Predicted category: {category}"

    if not include_llm_analysis:
        return prediction, "LLM analysis was skipped."

    try:
        return prediction, ANALYZER.analyze_with_groq(text)
    except Exception as exc:  # Gradio should show friendly setup errors.
        return prediction, f"LLM analysis unavailable: {exc}"


demo = gr.Interface(
    fn=classify_news,
    inputs=[
        gr.Textbox(
            label="News text",
            lines=6,
            placeholder="Paste a news headline or short article here.",
        ),
        gr.Checkbox(label="Include Groq LLM analysis", value=False),
    ],
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Analysis"),
    ],
    title="AG News Classification Demo",
    description=STATUS,
    examples=[
        ["Oil prices rose after major producers announced new supply cuts.", False],
        ["The national team won the final after a late goal in extra time.", False],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
    )
