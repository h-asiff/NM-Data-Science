# NM-Data-Science

emotion-analysis/
├── data/
│   └── emotions.csv
├── notebooks/
│   └── eda_model.ipynb
├── app/
│   └── app.py
├── requirements.txt
└── README.md

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('../data/emotions.csv')

# Basic preprocessing
X = df['text']
y = df['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')

import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Emotion Detection from Text")
user_input = st.text_area("Enter a sentence:")

if st.button("Predict Emotion"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        st.success(f"Predicted Emotion: {prediction[0]}")

import joblib
joblib.dump(model, '../app/model.pkl')
joblib.dump(tfidf, '../app/vectorizer.pkl')

pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
joblib

# Decoding Emotions Through Sentiment Analysis

This project detects emotions in text using machine learning. Trained on the Kaggle Emotion Dataset.

## Features
- Preprocessing with TF-IDF
- Logistic Regression model
- Streamlit web app

## Run the App
```bash
streamlit run app/app.py


---

Would you like me to bundle this into a downloadable `.zip` or create a GitHub repo structure you can directly clone?

