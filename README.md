# AG News Classification and Analysis

This project classifies news into four AG News categories: World, Sports,
Business, and Sci/Tech. It includes a clean machine-learning pipeline, an
optional deep-learning model, optional Groq/RAG explanation helpers, a structured
notebook, and a Gradio demo app.

## Features

- Text cleaning and reusable data-loading utilities
- TF-IDF + Logistic Regression classifier
- Optional Keras neural-network classifier
- Optional Groq LLM explanation
- Optional FAISS/LangChain RAG context retrieval
- Colab-ready Gradio demo

## Project Structure

```text
.
|-- app.py                  # Gradio demo application
|-- data_loader.py          # Dataset loading helpers
|-- dl_model.py             # Keras deep-learning model
|-- explanation.md          # Simple concept explanation
|-- ml_model.py             # TF-IDF + Logistic Regression model
|-- rag_pipeline.py         # Optional Groq and RAG helpers
|-- requirements.txt        # Python dependencies
|-- README.md               # Setup and usage guide
|-- text_preprocessing.py   # Text cleaning function
`-- DS_Project (1).ipynb    # Structured project notebook
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project can load AG News in two ways:

1. If `train.csv` and `test.csv` are in the project folder, it uses those files.
2. Otherwise, it downloads the `ag_news` dataset from Hugging Face.

Optional Groq analysis requires an API key:

```bash
set GROQ_API_KEY=your_key_here
```

On macOS/Linux, use:

```bash
export GROQ_API_KEY=your_key_here
```

You can also copy `.env.example` to `.env` and put your key there:

```bash
GROQ_API_KEY=your_key_here
```

## Train and Evaluate the ML Model

```python
from data_loader import load_ag_news
from ml_model import evaluate_ml_model, save_model, train_ml_model

train_df, test_df = load_ag_news(train_limit=12000, test_limit=2000)
model = train_ml_model(train_df)
metrics = evaluate_ml_model(model, test_df)
print(metrics["accuracy"])
print(metrics["classification_report"])
save_model(model)
```

## Predict One News Item

```python
from ml_model import load_model, predict_category

model = load_model("ml_model.joblib")
label_id, label_name = predict_category(
    model,
    "The company reported strong quarterly earnings after a rise in sales.",
)
print(label_id, label_name)
```

## Run the Gradio Demo

```bash
python app.py
```

For a public Colab link, run:

```bash
set GRADIO_SHARE=true
python app.py
```

In Colab, install dependencies, upload or clone the project, optionally set
`GROQ_API_KEY`, and run:

```python
!pip install -r requirements.txt
!GRADIO_SHARE=true python app.py
```

## Notebook

Open `DS_Project (1).ipynb` and run the cells from top to bottom. The notebook is
code-focused and follows this workflow:

1. Install/import dependencies
2. Load AG News
3. Clean text
4. Train and evaluate the ML model
5. Train and evaluate the optional DL model
6. Run optional Groq/RAG analysis
7. Launch the Gradio demo

## GitHub Submission

This folder is not currently initialized as a Git repository. To publish it:

```bash
git init
git add .
git commit -m "complete ag news classification project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

After pushing, submit your public repository URL.

## Notes

- Do not commit API keys. Use `GROQ_API_KEY` instead.
- `train.csv`, `test.csv`, saved models, and local environment files are ignored
  by `.gitignore`.
- Full concept explanation is in `explanation.md`.
