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
from scipy.sparse import csr_matrix

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
download_nltk_resources()

class SVM:
    def __init__(self, lambda_param=1e-4, epoch=1000, batch_size=256, tol=1e-4, random_state=42):
        self.lambda_param = lambda_param
        self.epoch = epoch
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self.is_trained = False

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = csr_matrix(X)
        
        self.num_samples, self.num_features = X.shape

        y_unique = np.unique(y)
        if len(y_unique) != 2:
            raise ValueError("Phân loại nhị phân cần 2 nhãn")
        if set(y_unique) == {0, 1}:
            y = np.where(y == 0, -1, 1)
        
        self.w = np.zeros(self.num_features, dtype=np.float32)
        self.b = 0.0

        np.random.seed(self.random_state)
        t = 0
        previous_objective = float("inf")

        for ep in range(1, self.epoch + 1):
            indices = np.random.permutation(self.num_samples)
            for start in range(0, self.num_samples, self.batch_size):
                t += 1
                end = start + self.batch_size
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                eta = 1.0 / (self.lambda_param * t)
                margins = y_batch * (X_batch.dot(self.w) + self.b)
                mask = margins < 1
                self.w *= (1 - eta * self.lambda_param)
                if np.any(mask):
                    X_violate = X_batch[mask]
                    y_violate = y_batch[mask]
                    self.w += (eta / self.batch_size) * np.dot(y_violate, X_violate.toarray() if hasattr(X_violate, "toarray") else X_violate)
                    self.b += (eta / self.batch_size) * np.sum(y_violate)
                norm_w = np.linalg.norm(self.w)
                factor = min(1, (1.0 / np.sqrt(self.lambda_param)) / (norm_w))
                self.w *= factor

            decision = X.dot(self.w) + self.b
            hinge_losses = np.maximum(0, 1 - y * decision)
            objective = 0.5 * self.lambda_param * np.dot(self.w, self.w) + np.mean(hinge_losses)
            
            if ep % 10 == 0:
                print(f"Epoch {ep}, Giá trị hàm mục tiêu: {objective:.4f}")
            
            if abs(previous_objective - objective) < self.tol:
                print(f"Dừng sớm tại epoch {ep}, giá trị hàm mục tiêu thay đổi: {abs(previous_objective - objective):.6f}")
                break
            previous_objective = objective

        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Mô hình chưa được huấn luỵen")
            
        if hasattr(X, "toarray"):
            X = csr_matrix(X)
            
        decision = X.dot(self.w) + self.b
        return np.where(decision >= 0, 1, 0)

@st.cache_resource
def load_model():
    try:
        with open('linear_svm.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
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