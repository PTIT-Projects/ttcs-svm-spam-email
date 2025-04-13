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

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
download_nltk_resources()

class LinearSVM:
    def __init__(self, C=1.0, max_iter=1000, lr=0.001, tolerance=1e-5):
        self.C = C
        self.max_iter = max_iter
        self.lr = lr
        self.tolerance = tolerance
        self.w = None
        self.b = 0

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        y_binary = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)

        alpha = np.zeros(n_samples)
        K = np.dot(X, X.T) * np.outer(y_binary, y_binary)
        for iteration in range(self.max_iter):
            alpha_prev = alpha.copy()

            margins = 1 - K.dot(alpha)

            mask = margins > 0
            alpha[mask] += self.lr * margins[mask]

            alpha = np.clip(alpha, 0, self.C)

            if np.max(np.abs(alpha - alpha_prev)) < self.tolerance:
                break
        self.w = np.dot(X.T, alpha * y_binary)

        sv_indices = alpha > 1e-5
        if np.any(sv_indices):
            self.b = np.mean(y_binary[sv_indices] - np.dot(X[sv_indices], self.w))

    def predict(self, X):
        """Predict class labels for samples in X."""
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, 0)

    def get_parameters(self):
      print(f'w: {self.w}')
      print(f'b: {self.b}')

    def decision_function(self, X):
        """Return distance of samples to the decision boundary."""
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        return np.dot(X, self.w) + self.b

@st.cache_resource
def load_model():
    try:
        with open('linear_svm.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('hashing_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Không tìm thấy file model/vectorizer")
        return None, None

model, vectorizer = load_model()

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
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text) if isinstance(text, str) else text
    text = re.sub(r'^\s+|\s+$', '', text).strip() if isinstance(text, str) else text
    text = remove_html_xml(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = remove_emails(text)
    tokens = word_tokenize(text)
    ENGLISH_STOP_WORDS = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


st.title("Demo phân loại email tiếng Anh spam ")


user_input = st.text_area("Nhập nội dung email:", height=200)

if st.button("Kiểm tra"):
    if not user_input:
        st.warning("Hãy nhập nội dung để phân tích")
    elif model is None or vectorizer is None:
        st.error("Model/vectorizer không load được")
    else:
        with st.spinner("Đang phân tích..."):
            preprocessed_text = preprocess_text(user_input)
            if hasattr(vectorizer, 'transform'):
                features = vectorizer.transform([preprocessed_text])
            else:
                st.error("Không tìm thấy vectorizer")
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.error("🚨 Email có khả năng là SPAM")
            else:
                st.success("✅Email có khả năng không phải là SPAM ")
            
            st.write("### Văn bản sau khi được tiền xử lý:")
            st.write(preprocessed_text)