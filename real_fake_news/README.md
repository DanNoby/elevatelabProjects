# Fake News Detection Web App (Project 6)

This project is a machine learning-based web application that classifies news articles as **Fake** or **Real**.

## Features

- Classifies user-input news articles as **Fake** or **Real**
- Provides an explanation showing top contributing words
- Streamlit-powered interactive web interface
- Supports testing with sample inputs via sidebar
- Cleaned and vectorized text using NLTK + TF-IDF
- Trained with Multinomial Naive Bayes

## Tools & Libraries

- Python
- Pandas, Sklearn, NLTK
- TfidfVectorizer
- MultinomialNB / LogisticRegression
- Streamlit (for UI)

## 💻 Running the App

- Install dependencies and Launch web app:
  ```bash
  pip install -r requirements.txt
  streamlit run streamlitapp.py
  ```
