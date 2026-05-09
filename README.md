# AG News Classification and Analysis

A complete Python project for classifying short news articles into the four AG
News categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

The project includes a traditional machine-learning model, an optional
deep-learning model, optional Groq LLM analysis, optional RAG retrieval, a
structured Jupyter Notebook, and a Gradio web demo.

Repository:
https://github.com/ahmedokasha74/ag-news-classification

## Project Highlights

- Clean, modular Python code
- TF-IDF + Logistic Regression classifier
- Optional Keras deep-learning classifier
- Text preprocessing pipeline
- Optional Groq LLM explanation
- Optional FAISS/LangChain RAG pipeline
- Gradio demo app for interactive testing
- Colab-ready notebook workflow
- Simple concept explanation in `explanation.md`

## Dataset

The project uses the AG News dataset, which contains four categories:

- World
- Sports
- Business
- Sci/Tech

The loader works in two ways:

- If `train.csv` and `test.csv` exist locally, it uses them.
- Otherwise, it downloads the `ag_news` dataset from Hugging Face.

## Project Structure

```text
.
|-- app.py                  # Gradio web demo
|-- data_loader.py          # Dataset loading and preparation
|-- dl_model.py             # Optional TensorFlow/Keras model
|-- explanation.md          # Simple written concept explanation
|-- ml_model.py             # TF-IDF + Logistic Regression model
|-- rag_pipeline.py         # Optional Groq and RAG helpers
|-- requirements.txt        # Project dependencies
|-- README.md               # Project documentation
|-- text_preprocessing.py   # Text cleaning utilities
`-- DS_Project (1).ipynb    # Structured project workflow notebook
```

## Setup

Clone the repository:

```bash
git clone https://github.com/ahmedokasha74/ag-news-classification.git
cd ag-news-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The app can run without an LLM key if you only want classification.

For optional Groq LLM analysis, create a `.env` file:

```bash
cp .env.example .env
```

Then add your Groq key:

```env
GROQ_API_KEY=your_groq_key_here
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
MODEL_PATH=ml_model.joblib
TRAIN_LIMIT=12000
TEST_LIMIT=2000
```

The `.env` file is ignored by Git, so private keys are not uploaded.

## Run the Gradio Demo

Start the web app:

```bash
python app.py
```

Open the local URL shown in the terminal:

```text
http://127.0.0.1:7860
```

Enter a news headline or paragraph and submit it. If you only need the category
prediction, leave **Include Groq LLM analysis** unchecked.

Example input:

```text
Apple reported stronger-than-expected quarterly earnings as demand for its iPhone and cloud services increased across global markets.
```

Expected category:

```text
Business
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
    "The national team won the championship after scoring a late goal.",
)

print(label_id, label_name)
```

## Optional LLM Analysis

After setting `GROQ_API_KEY`, enable the checkbox in the Gradio app. The app will
return:

- Predicted category
- Explanation of why the category fits
- Short summary
- Key entities
- Sentiment description

## Notebook Workflow

Open `DS_Project (1).ipynb` and run the cells from top to bottom.

The notebook covers:

1. Loading dependencies
2. Loading AG News
3. Cleaning text
4. Training and evaluating the ML model
5. Training and evaluating the optional DL model
6. Running optional Groq/RAG analysis
7. Launching the Gradio demo

## Colab Usage

In Google Colab:

```python
!git clone https://github.com/ahmedokasha74/ag-news-classification.git
%cd ag-news-classification
!pip install -r requirements.txt
!python app.py
```

For a public Gradio link, set:

```python
import os
os.environ["GRADIO_SHARE"] = "true"
!python app.py
```

## Notes

- Do not commit `.env`, API keys, saved models, or downloaded datasets.
- `ml_model.joblib`, `train.csv`, and `test.csv` are ignored by `.gitignore`.
- The written concept explanation is available in `explanation.md`.

