import pandas as pd
import numpy as np
import nltk
import spacy
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF resume
def extract_resume_text(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to preprocess resume text
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatization & stopword removal
    return " ".join(tokens)

# Sample dataset for training model (for simplicity, real dataset needed)
resume_samples = [
    "Python Machine Learning Data Science SQL",
    "Marketing SEO Content Writing Social Media",
    "Java Backend Development Spring Hibernate",
    "Deep Learning NLP TensorFlow PyTorch"
]
labels = ["Data Science", "Marketing", "Software Development", "AI/ML"]  # Example job categories

# TF-IDF Vectorization & Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resume_samples)
model = LogisticRegression()
model.fit(X, labels)

# Streamlit Web App
def main():
    st.title("AI-Powered Resume Analyzer")
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        text = extract_resume_text(uploaded_file)
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        st.subheader("Predicted Job Fit:")
        st.write(f"🔹 {prediction}")

if __name__ == "__main__":
    main()
