# SVM Email Spam Detector

A Streamlit web application that detects spam emails using a Support Vector Machine (SVM) model.

## Features

- Web interface for entering email content
- Real-time spam detection using SVM
- Text preprocessing including tokenization, stemming, and stopword removal
- Docker containerization for easy deployment

## Prerequisites

- Docker installed on your system
- The trained model (`linear_svm.pkl`) and vectorizer (`hashing_vectorizer.pkl`) files

## Getting Started

### Building the Docker Image

1. Make sure the following files are in the same directory:
   - `app.py` (Streamlit application)
   - `Dockerfile`
   - `requirements.txt`
   - `linear_svm.pkl` (your trained SVM model)
   - `hashing_vectorizer.pkl` (your trained vectorizer)

2. Build the Docker image:
   ```bash
   docker build -t spam-detector .
   ```

3. Run the container:
   ```bash
   docker run -p 8501:8501 spam-detector
   ```

4. Access the application in your web browser at:
   ```
   http://localhost:8501
   ```

## Using the Application

1. Enter or paste email content into the text area
2. Click "Check if Spam" button
3. View the prediction result and preprocessed text

## Model Training

The SVM model used in this application was trained on a dataset of spam and non-spam emails. The training code is available in the `svm.py` and `svm.ipynb` files in this repository.