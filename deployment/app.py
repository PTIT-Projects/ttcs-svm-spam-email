import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import HashingVectorizer

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    
download_nltk_resources()

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('linear_svm.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('hashing_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer files not found! Please ensure the pickle files are in the correct location.")
        return None, None

model, vectorizer = load_model()

# Text preprocessing functions
def remove_html_xml(text):
    try:
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    except:
        return text

def remove_special_characters(word):
    return word.translate(str.maketrans('', '', string.punctuation))

def remove_urls(text):
    return re.sub(r'http\S+|www\S+|\S+\.(com|net|org|edu|gov|mil|int|info|biz|co)\S+', '', text)

def remove_emails(text):
    return re.sub(r'\S+@\S+', '', text)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text) if isinstance(text, str) else text
    
    # Remove whitespace
    text = re.sub(r'^\s+|\s+$', '', text).strip() if isinstance(text, str) else text
    
    # Remove HTML/XML
    text = remove_html_xml(text)
    
    # Remove special characters
    text = remove_special_characters(text)
    
    # Remove URLs
    text = remove_urls(text)
    
    # Remove email addresses
    text = remove_emails(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    ENGLISH_STOP_WORDS = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back to string
    return ' '.join(tokens)

# Streamlit UI
st.title("Email Spam Detector")
st.write("This application uses a Support Vector Machine (SVM) model to predict whether an email is spam or not.")

user_input = st.text_area("Enter the email content:", height=200)

if st.button("Check if Spam"):
    if not user_input:
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded correctly. Please check the console for errors.")
    else:
        with st.spinner("Analyzing text..."):
            # Preprocess the text
            preprocessed_text = preprocess_text(user_input)
            
            # Vectorize the text
            if hasattr(vectorizer, 'transform'):
                features = vectorizer.transform([preprocessed_text])
            else:
                # If vectorizer is not found, fallback to a new HashingVectorizer
                fallback_vectorizer = HashingVectorizer(n_features=1000)
                features = fallback_vectorizer.transform([preprocessed_text])
            
            # Make prediction
            prediction = model.predict(features)
            
            # Display result
            if prediction[0] == 1:
                st.error("🚨 This message appears to be SPAM!")
            else:
                st.success("✅ This message appears to be legitimate (NOT spam).")
            
            st.write("### Preprocessed Text:")
            st.write(preprocessed_text)