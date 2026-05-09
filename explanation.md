# AG News Classification: Simple Concept Explanation

This project teaches a computer to read a short news article and decide which
topic it belongs to: World, Sports, Business, or Sci/Tech.

## Main Idea

News stories usually contain words that give clues about their category. For
example, a sports story may mention a team, match, score, or championship. A
business story may mention stocks, companies, prices, or markets.

The project uses those word clues to train models that can recognize patterns in
new articles.

## How The Project Works

1. The AG News dataset is loaded.
2. Each article is cleaned by removing links, punctuation, numbers, and extra
   spaces.
3. A machine-learning model turns text into numbers using TF-IDF.
4. Logistic Regression learns which word patterns match each news category.
5. A small neural-network model is included as a deep-learning comparison.
6. Optional Groq analysis explains the prediction in simple language.
7. Optional RAG retrieves similar news examples before asking the LLM to explain.
8. A Gradio app lets users paste their own news text and test the classifier.

## TF-IDF In Simple Words

TF-IDF gives higher importance to words that are useful for identifying a
document. Common words like "the" and "and" are not very helpful, so they get low
importance. More meaningful words like "election", "tournament", "market", or
"software" can receive higher importance.

## Logistic Regression In Simple Words

Logistic Regression is a simple classification model. It looks at the word
scores from TF-IDF and chooses the category that best matches the article.

## Deep Learning In Simple Words

The neural network turns words into small numeric representations called
embeddings. It then learns patterns from those representations and predicts one
of the four categories.

## RAG In Simple Words

RAG means Retrieval-Augmented Generation. Before the LLM explains an article, the
system first searches for similar examples from the dataset. The LLM can use
those examples as extra context, which makes the explanation more grounded.

## Why This Project Is Useful

This project shows a complete text-classification workflow: data loading,
cleaning, model training, evaluation, optional LLM explanation, and deployment
with a simple web interface.

